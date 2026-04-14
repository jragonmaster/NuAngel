// NeuraCell SC2 Hook — Inference Runner
// ============================================================
// Architecture: NeuraCellNet (train.py v3) + Option B memory system
//
// OUTPUT CHANGE (v3): Replaced single categorical(16) command head with
// independent per-output sigmoid heads:
//
//   BUTTON HEAD  (7 neurons, sigmoid):
//     [0] A   [1] B   [2] X   [3] Y   [4] L   [5] Z   [6] Shoulder
//     Each fires independently at threshold > BTN_THRESHOLD (0.5).
//     The model can now press A+B simultaneously, jump while attacking, etc.
//
//   MEM HEAD  (3 neurons, sigmoid ? 3-bit int):
//     Encodes which memory operation fires this frame (0 = none, 1-7 = op).
//     Independent of buttons — mem ops and button presses can co-occur.
//     0b000 (all < 0.5) = no mem op
//     0b001 = MEM_WRITE   0b010 = MEM_READ    0b011 = MEM_QUERY
//     0b100 = MEM_CONSOLIDATE  0b101 = MEM_BWRITE  0b110 = MEM_BQUERY
//     0b111 = RESET_SELF
//
//   PARAMS HEAD  (2 neurons, tanh ? [-1,1]):  stick X, stick Y  (unchanged)
//   PATCH HEAD   (32 neurons, tanh×0.1):       inner net patch   (unchanged)
//   SELF_PRED    (8 neurons, linear):           self-model target (unchanged)
//
// State vector (24-dim, unchanged):
//  [0]  mem_read_result   [1]  mem_hit         [2]  conseq_acted
//  [3]  conseq_reset      [4]  reserved        [5]  conseq_retrieved
//  [6]  stock_delta       [7]  damage_delta    [8]  p1_pct
//  [9]  p2_pct            [10] mem_occupancy   [11] mem_health
//  [12-19] last_self_pred (model's raw self-prediction, not inner_out)
//  [20] time_in_stock     [21] self_pred_error [22] dist_norm
//  [23] facing
//
// Build: link torch, ViGEmClient, psapi
// Run:   place neura_cell.pt in same dir, run as Admin

#include <torch/script.h>
#include <torch/torch.h>
#include <windows.h>
#include <tlhelp32.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <array>
#include <string>
#include "ViGEmClient.h"
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "ViGEmClient.lib")


static HANDLE hMapFile = NULL;
static LPVOID pBuf = NULL;
static const char* BUFFER_PATH = "replay_buffer.bin";
static const uint32_t BUFFER_CAPACITY = 8192;
static uint32_t write_head = 0;
static uint64_t total_written = 0;   // use 64-bit to avoid overflow

// ?????????????????????????????????????????????????????????????????????????????
// Shared Replay Buffer (for Python trainer) — 45 floats per experience - must match Python exactly
// ?????????????????????????????????????????????????????????????????????????????

static const uint32_t EXP_FLOATS = 45;

// Previous p1/p2 percentages for reward delta calculation
static float prev_p1_pct = 0.0f;
static float prev_p2_pct = 0.0f;
static constexpr int   SEQ_LEN = 64;
static constexpr int STATE_DIM = 26;        // was 24, now +2 for sin/cos
static constexpr int FLAT_DIM = SEQ_LEN * STATE_DIM; // will become 1664
static constexpr int   INNER_N = 32;
static constexpr int   MEM_SLOTS = 16;
static constexpr int   MEM_VALS = 8;
static constexpr float PATCH_SCALE = 0.03f;
static constexpr float WEIGHT_CLAMP = 1.0f;
static constexpr float BTN_THRESHOLD = 0.5f;

// Button head indices
static constexpr int BTN_A = 0;
static constexpr int BTN_B = 1;
static constexpr int BTN_X = 2;
static constexpr int BTN_Y = 3;
static constexpr int BTN_L = 4;
static constexpr int BTN_Z = 5;
static constexpr int BTN_SHOULDER = 6;

static const char* kMemOp[8] = {
    "NONE", "MEM_WRITE", "MEM_READ", "MEM_QUERY",
    "MEM_CONSOLIDATE", "MEM_BWRITE", "MEM_BQUERY", "RESET_SELF"
};

// ?????????????????????????????????????????????????????????????????????????????
//  Dolphin memory addresses
// ?????????????????????????????????????????????????????????????????????????????
namespace Dolphin {
    const uintptr_t BASE = 0x7FFF0000;
    const uint32_t  ADDR_P1_HP = 0x0034EA1C;
    const uint32_t  ADDR_P2_HP = 0x0036FABC;
    const uint32_t  ADDR_P1X = 0x0034FDE0;
    const uint32_t  ADDR_P1Y = 0x0034FDE8;
    const uint32_t  ADDR_P2X = 0x00370E80;
    const uint32_t  ADDR_P2Y = 0x00370E88;
    const uint32_t  ADDR_FACING = 0x0034FD34;
    const uint32_t  ADDR_P1_STKS = 0x0034EA20;
    const uint32_t  ADDR_P2_STKS = 0x0036FAC0;
}

// ?????????????????????????????????????????????????????????????????????????????
//  Dolphin memory reader
// ?????????????????????????????????????????????????????????????????????????????
class MemoryReader {
    HANDLE    hProc = NULL;
    uintptr_t base = Dolphin::BASE;
public:
    bool Connect() {
        PROCESSENTRY32W pe{ sizeof(pe) };
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (Process32FirstW(snap, &pe)) {
            do {
                if (wcsstr(pe.szExeFile, L"Dolphin.exe") ||
                    wcsstr(pe.szExeFile, L"DolphinQt2.exe")) {
                    hProc = OpenProcess(PROCESS_VM_READ, FALSE, pe.th32ProcessID);
                    CloseHandle(snap);
                    return hProc != NULL;
                }
            } while (Process32NextW(snap, &pe));
        }
        CloseHandle(snap);
        return false;
    }

    float ReadF32(uint32_t off) {
        uint8_t buf[4];
        if (!ReadProcessMemory(hProc, (LPCVOID)(base + off), buf, 4, NULL)) return 0.f;
        uint8_t sw[4] = { buf[3], buf[2], buf[1], buf[0] };
        float f; memcpy(&f, sw, 4); return f;
    }

    int ReadI32(uint32_t off) {
        uint8_t buf[4];
        if (!ReadProcessMemory(hProc, (LPCVOID)(base + off), buf, 4, NULL)) return 0;
        uint8_t sw[4] = { buf[3], buf[2], buf[1], buf[0] };
        int v; memcpy(&v, sw, 4); return v;
    }
};

bool InitReplayBuffer() {
    // Create or open the file
    HANDLE hFile = CreateFileA(BUFFER_PATH, GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
        OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "[Buffer] Failed to create/open replay_buffer.bin\n";
        return false;
    }

    // Extend file to proper size
    LARGE_INTEGER fileSize;
    fileSize.QuadPart = 8ULL + (uint64_t)BUFFER_CAPACITY * 45 * 4;  // header + 8192 * 45 floats
    SetFilePointerEx(hFile, fileSize, NULL, FILE_BEGIN);
    SetEndOfFile(hFile);

    hMapFile = CreateFileMappingA(hFile, NULL, PAGE_READWRITE, 0, 0, NULL);
    CloseHandle(hFile);
    if (!hMapFile) {
        std::cerr << "[Buffer] CreateFileMapping failed\n";
        return false;
    }

    pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!pBuf) {
        std::cerr << "[Buffer] MapViewOfFile failed\n";
        return false;
    }

    std::cout << "[NeuraCell] Replay buffer initialized (8192 slots)\n";
    return true;
}

// Call this every frame after you have all the values
void WriteExperience(const float* state,
    bool bA, bool bB, bool bX, bool bY, bool bL, bool bZ, bool bSh,
    int mem_op,
    float param0, float param1,
    const torch::Tensor& btn_logits,   // for log probs
    const torch::Tensor& mem_logits,
    bool done = false,
    float p1_prev = 0.f, float p2_prev = 0.f) {

    if (!pBuf){ printf("[Buffer] Not initialized!\n"); return; }

    float* slot = (float*)((uint8_t*)pBuf + 8 + (write_head * 45 * 4));  // 45 floats per exp

    // 1. State (24)
    memcpy(slot + 0, state, 26 * sizeof(float));

    // 2. Button actions (7 binary)
    slot[24] = bA ? 1.0f : 0.0f;
    slot[25] = bB ? 1.0f : 0.0f;
    slot[26] = bX ? 1.0f : 0.0f;
    slot[27] = bY ? 1.0f : 0.0f;
    slot[28] = bL ? 1.0f : 0.0f;
    slot[29] = bZ ? 1.0f : 0.0f;
    slot[30] = bSh ? 1.0f : 0.0f;

    // 3. Mem op
    slot[31] = (float)mem_op;

    // 4. Params
    slot[32] = param0;
    slot[33] = param1;

    // 5. Button log probs (approximate from logits)
    auto btn_logits_adj = btn_logits - 2.0;
    auto btn_probs = torch::sigmoid(btn_logits_adj);

    for (int i = 0; i < 7; ++i) {
        float p = btn_probs[i].item<float>();
        p = std::clamp(p, 1e-6f, 1.0f - 1e-6f);
        slot[34 + i] = std::log(p / (1.0f - p));   // logit ? log prob for Bernoulli
    }

    // 6. Mem log prob (3 bits)
    auto mem_probs = torch::sigmoid(mem_logits);
    float mem_logp = 0.0f;
    for (int i = 0; i < 3; ++i) {
        float p = mem_probs[i].item<float>();
        p = std::clamp(p, 1e-6f, 1.0f - 1e-6f);
        mem_logp += std::log(((mem_op >> i) & 1) ? p : (1.0f - p));
    }
    slot[41] = mem_logp;

    // 7. Done + prev hp
    slot[42] = done ? 1.0f : 0.0f;
    slot[43] = p1_prev;
    slot[44] = p2_prev;

    // Update header
    *(uint32_t*)pBuf = write_head;                    // write_head
    *((uint32_t*)pBuf + 1) = (uint32_t)total_written; // total_written (low 32)

    write_head = (write_head + 1) % BUFFER_CAPACITY;
    total_written++;

    // Optional: flush less often
    if (total_written % 64 == 0)
        FlushViewOfFile(pBuf, 0);
}


// ?????????????????????????????????????????????????????????????????????????????
//  ViGEm virtual controller
// ?????????????????????????????????????????????????????????????????????????????
class VirtualController {
    PVIGEM_CLIENT client = nullptr;
    PVIGEM_TARGET pad = nullptr;
    XUSB_REPORT   report = {};
public:
    bool Connect() {
        client = vigem_alloc();
        if (!VIGEM_SUCCESS(vigem_connect(client))) {
            std::cerr << "[ViGEm] Bus connection failed.\n"; return false;
        }
        pad = vigem_target_x360_alloc();
        if (!VIGEM_SUCCESS(vigem_target_add(client, pad))) {
            std::cerr << "[ViGEm] Add controller failed.\n"; return false;
        }
        std::cout << "[ViGEm] Xbox 360 controller connected.\n";
        return true;
    }

    void Send(float sX, float sY,
        bool bA, bool bB, bool bX, bool bY,
        bool bL, bool bZ, bool bShoulder) {
        XUSB_REPORT_INIT(&report);
        report.sThumbLX = (SHORT)std::clamp(sX * 32767.f, -32767.f, 32767.f);
        report.sThumbLY = (SHORT)std::clamp(sY * 32767.f, -32767.f, 32767.f);
        if (bA)        report.wButtons |= XUSB_GAMEPAD_A;
        if (bB)        report.wButtons |= XUSB_GAMEPAD_B;
        if (bX)        report.wButtons |= XUSB_GAMEPAD_X;
        if (bY)        report.wButtons |= XUSB_GAMEPAD_Y;
        if (bL)        report.wButtons |= XUSB_GAMEPAD_LEFT_SHOULDER;
        if (bZ)        report.wButtons |= XUSB_GAMEPAD_RIGHT_SHOULDER;
        if (bShoulder) report.bLeftTrigger = 255;
        vigem_target_x360_update(client, pad, report);
    }

    void Disconnect() {
        if (pad) { vigem_target_remove(client, pad); vigem_target_free(pad); }
        if (client) { vigem_disconnect(client); vigem_free(client); }
    }
    ~VirtualController() { Disconnect(); }
};

// ?????????????????????????????????????????????????????????????????????????????
//  Inner net (32×32, Leaky ReLU)
// ?????????????????????????????????????????????????????????????????????????????
float gInnerW[INNER_N][INNER_N] = {};
float gInnerB[INNER_N] = {};

static void run_inner(const float* in, float* out) {
    for (int i = 0; i < INNER_N; ++i) {
        float v = gInnerB[i];
        for (int j = 0; j < INNER_N; ++j) v += gInnerW[i][j] * in[j];
        out[i] = (v > 0.f) ? v : v * 0.01f;  // leaky ReLU
    }
}

static void consolidate_inner(int iters = 8) {
    float buf[INNER_N] = {}, out[INNER_N] = {};
    for (int k = 0; k < iters; ++k) { run_inner(buf, out); memcpy(buf, out, sizeof(buf)); }
}

static float inner_health() {
    float s = 0.f;
    for (int i = 0; i < INNER_N; ++i) {
        s += gInnerB[i] * gInnerB[i];
        for (int j = 0; j < INNER_N; ++j) s += gInnerW[i][j] * gInnerW[i][j];
    }
    return 1.f / (1.f + s * 0.01f);
}

// ?????????????????????????????????????????????????????????????????????????????
//  Tactical working memory — 16 slots × 8 values
// ?????????????????????????????????????????????????????????????????????????????
float gMemSlots[MEM_SLOTS][MEM_VALS] = {};
bool  gMemWritten[MEM_SLOTS] = {};

static int mem_addr(float p) {
    return (int)std::clamp((p * 0.5f + 0.5f) * MEM_SLOTS, 0.f, (float)(MEM_SLOTS - 1));
}

static float mem_occupancy() {
    int n = 0; for (int i = 0; i < MEM_SLOTS; ++i) if (gMemWritten[i]) ++n;
    return (float)n / MEM_SLOTS;
}

static float mem_query(float query_val, float* inner_out_buf) {
    float best_d = 1e9f; int best_s = -1;
    for (int s = 0; s < MEM_SLOTS; ++s) {
        if (!gMemWritten[s]) continue;
        float d = std::abs(gMemSlots[s][0] - query_val);
        if (d < best_d) { best_d = d; best_s = s; }
    }
    if (best_s < 0) return 0.f;
    float in_buf[INNER_N] = {};
    for (int j = 0; j < MEM_VALS; ++j) in_buf[j] = gMemSlots[best_s][j];
    float out_buf[INNER_N] = {};
    run_inner(in_buf, out_buf);
    if (inner_out_buf)
        for (int j = 0; j < 8; ++j) inner_out_buf[j] = out_buf[j];
    return std::clamp(1.f - best_d, 0.f, 1.f);
}

// ?????????????????????????????????????????????????????????????????????????????
//  State persistence
// ?????????????????????????????????????????????????????????????????????????????
static void save_state(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "[NeuraCell] Save failed.\n"; return; }
    f.write(reinterpret_cast<const char*>(gInnerW), sizeof(gInnerW));
    f.write(reinterpret_cast<const char*>(gInnerB), sizeof(gInnerB));
    f.write(reinterpret_cast<const char*>(gMemSlots), sizeof(gMemSlots));
    f.write(reinterpret_cast<const char*>(gMemWritten), sizeof(gMemWritten));
    std::cout << "[NeuraCell] State saved -> " << path << "\n";
}

static void load_state(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cout << "[NeuraCell] No save file found.\n"; return; }
    f.read(reinterpret_cast<char*>(gInnerW), sizeof(gInnerW));
    f.read(reinterpret_cast<char*>(gInnerB), sizeof(gInnerB));
    f.read(reinterpret_cast<char*>(gMemSlots), sizeof(gMemSlots));
    f.read(reinterpret_cast<char*>(gMemWritten), sizeof(gMemWritten));
    std::cout << "[NeuraCell] State loaded <- " << path << "\n";
}

// ?????????????????????????????????????????????????????????????????????????????
//  Helpers
// ?????????????????????????????????????????????????????????????????????????????
static void apply_patch(const torch::Tensor& patch, float scale = PATCH_SCALE) {
    for (int i = 0; i < INNER_N; ++i) {
        float delta = patch[i].item<float>() * scale;
        int wi = i % INNER_N;
        int wj = (i * 7 + 3) % INNER_N;
        gInnerW[wi][wj] = std::clamp(gInnerW[wi][wj] + delta, -WEIGHT_CLAMP, WEIGHT_CLAMP);
    }
}

// ?????????????????????????????????????????????????????????????????????????????
//  Main
// ?????????????????????????????????????????????????????????????????????????????
int main() {
    std::cout << "--- NeuraCell SC2 Hook v3 (per-button sigmoid) ---\n";
    std::cout << "SEQ_LEN=" << SEQ_LEN << "  STATE_DIM=" << STATE_DIM
        << "  FLAT=" << FLAT_DIM << "\n";

    torch::jit::script::Module model;
    try {
        model = torch::jit::load("neura_cell.pt");
        model.eval();
        std::cout << "[Model] neura_cell.pt loaded.\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "[Error] " << e.what() << "\n"; return -1;
    }

    MemoryReader      mem;
    VirtualController pad;
    if (!mem.Connect()) { std::cerr << "[Error] Dolphin not found.\n"; return -1; }
    if (!pad.Connect())  return -1;
    if (!InitReplayBuffer()) {
        std::cerr << "[Warning] Replay buffer init failed — trainer will use fallback.\n";
    }
    load_state("neura_cell_sc2.bin");

    float history[SEQ_LEN][STATE_DIM] = {};
    torch::NoGradGuard no_grad;

    float prev_sX = 0.f, prev_sY = 0.f;
    float self_pred_err = 0.5f;
    float mem_read_result = 0.f;
    float mem_hit = 0.f;
    float inner_out[8] = {};
    float last_self_pred[8] = {};
    float c_acted = 0.f, c_reset = 0.f, c_retrieved = 0.f;
    int   time_in_stock = 0;
    float prev_p1_stks = 4.f;
    int   neutral_streak = 0;
    int   frame = 0;

    // Mem op exploration — helps early training when mem head is near-zero
    int explore_mem_countdown = 0;
    int explore_mem_op = 0;

    float action_hint_sX = 0.f, action_hint_sY = 0.f;
    bool  action_hint_live = false;

    const float STICK_MOMENTUM = 0.59f;

    std::cout << "[NeuraCell] Running. Ctrl+C to stop.\n\n";

    try {
        while (true) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // ?? 1. Read game state ?????????????????????????????????????????????
            float p1_hp = mem.ReadF32(Dolphin::ADDR_P1_HP);
            float p2_hp = mem.ReadF32(Dolphin::ADDR_P2_HP);
            float p1x = mem.ReadF32(Dolphin::ADDR_P1X);
            float p1y = mem.ReadF32(Dolphin::ADDR_P1Y);
            float p2x = mem.ReadF32(Dolphin::ADDR_P2X);
            float p2y = mem.ReadF32(Dolphin::ADDR_P2Y);
            float facing = mem.ReadF32(Dolphin::ADDR_FACING);
            float p1_stks = (float)mem.ReadI32(Dolphin::ADDR_P1_STKS);
            float p2_stks = (float)mem.ReadI32(Dolphin::ADDR_P2_STKS);

            float dist = std::sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y));
            float p1_pct = std::clamp(p1_hp / 999.f, 0.f, 1.f);
            float p2_pct = std::clamp(p2_hp / 999.f, 0.f, 1.f);
            float stock_delta = std::clamp((p1_stks - p2_stks) / 4.f, -1.f, 1.f);
            float dmg_delta = std::clamp((p2_pct - p1_pct), -1.f, 1.f);
            float dist_norm = std::clamp(dist / 28.3f, 0.f, 1.f);
            float face_norm = (facing > 0.f) ? 1.f : -1.f;

            if (p1_stks != prev_p1_stks) {
                time_in_stock = 0;
                if (p1_stks < prev_p1_stks) {
                    std::fill(&gMemSlots[0][0], &gMemSlots[0][0] + MEM_SLOTS * MEM_VALS, 0.f);
                    std::fill(gMemWritten, gMemWritten + MEM_SLOTS, false);
                    std::cout << "[NeuraCell] Stock lost — memory cleared.\n";
                }
                prev_p1_stks = p1_stks;
            }
            ++time_in_stock;

            // ?? 2. Build state vector ?????????????????????????????????????????
            float state[STATE_DIM] = {};
            state[0] = mem_read_result;
            state[1] = mem_hit;
            state[2] = c_acted;
            state[3] = c_reset;
            state[4] = 0.f;
            state[5] = c_retrieved;
            state[6] = stock_delta;
            state[7] = dmg_delta;
            state[8] = p1_pct;
            state[9] = p2_pct;
            state[10] = mem_occupancy();
            state[11] = inner_health();

            for (int i = 0; i < 8; ++i)
                state[12 + i] = last_self_pred[i];

            state[20] = std::clamp((float)time_in_stock / 3600.f, 0.f, 1.f);
            state[21] = self_pred_err;
            state[22] = dist_norm;
            state[23] = face_norm * 0.5f + 0.5f;

            // ?? NEW: Frame phase (gives model awareness of current time) ?????
            const float phase_speed = 0.022f;        // tune this (0.01 = very slow, 0.05 = faster)
            state[24] = std::sin(frame * phase_speed);
            state[25] = std::cos(frame * phase_speed);

            // Reset transient flags
            mem_read_result = 0.f;
            mem_hit = 0.f;
            c_acted = c_reset = c_retrieved = 0.f;

            // ?? 3. Roll history ???????????????????????????????????????????????
            memmove(&history[0], &history[1], (SEQ_LEN - 1) * STATE_DIM * sizeof(float));
            memcpy(&history[SEQ_LEN - 1], state, STATE_DIM * sizeof(float));

            // ?? 4. Inference ??????????????????????????????????????????????????
            torch::Tensor input = torch::from_blob(
                history, { 1, FLAT_DIM }, torch::kFloat32).clone();

            auto out_tup = model.forward({ input }).toTuple();
            // v3 output layout:
            //   [0] btn_logits (7)   sigmoid ? A B X Y L Z Shoulder
            //   [1] mem_logits (3)   sigmoid ? 3-bit mem op index
            //   [2] params     (2)   tanh    ? stick X Y
            //   [3] patch      (32)  tanh×0.1 ? inner net patch
            //   [4] self_pred  (8)   linear  ? self-model target
            auto btn_logits = out_tup->elements()[0].toTensor()[0];  // (7,)
            auto mem_logits = out_tup->elements()[1].toTensor()[0];  // (3,)
            auto params = out_tup->elements()[2].toTensor()[0];  // (2,)
            auto patch = out_tup->elements()[3].toTensor()[0];  // (32,)
            auto self_pred = out_tup->elements()[4].toTensor()[0];  // (8,)

            for (int i = 0; i < 8; ++i)
                last_self_pred[i] = self_pred[i].item<float>();

            // ?? 5. Decode buttons (independent sigmoid per button) ?????????????
            
            auto btn_probs = torch::sigmoid(btn_logits);
            bool bA = btn_probs[BTN_A].item<float>() > BTN_THRESHOLD;
            bool bB = btn_probs[BTN_B].item<float>() > BTN_THRESHOLD;
            bool bX = btn_probs[BTN_X].item<float>() > BTN_THRESHOLD;
            bool bY = btn_probs[BTN_Y].item<float>() > BTN_THRESHOLD;
            bool bL = btn_probs[BTN_L].item<float>() > BTN_THRESHOLD;
            bool bZ = btn_probs[BTN_Z].item<float>() > BTN_THRESHOLD;
            bool bTrigger = btn_probs[BTN_SHOULDER].item<float>() > BTN_THRESHOLD;

            if (bA || bB || bX || bY || bL || bZ || bTrigger)
                c_acted = 1.f;

            // ?? 6. Decode mem op (3 sigmoid bits ? integer 0-7) ???????????????
            auto mem_probs = torch::sigmoid(mem_logits);
            int mem_bit0 = mem_probs[0].item<float>() > BTN_THRESHOLD ? 1 : 0;
            int mem_bit1 = mem_probs[1].item<float>() > BTN_THRESHOLD ? 1 : 0;
            int mem_bit2 = mem_probs[2].item<float>() > BTN_THRESHOLD ? 1 : 0;
            int mem_op = mem_bit0 | (mem_bit1 << 1) | (mem_bit2 << 2);

            // Exploration: 10% chance to override with random mem op for 8 frames
            {
                if (((float)rand() / RAND_MAX) < 0.1f && explore_mem_countdown <= 0) {
                    explore_mem_op = (rand() % 7) + 1;
                    explore_mem_countdown = 8;
                }
                if (explore_mem_countdown > 0) {
                    mem_op = explore_mem_op;
                    --explore_mem_countdown;
                }
            }

            // ?? 7. Improved Stick Output ???????????????????????????????????????
            // ?? 7. Improved Stick Output with Decimal Shift ?????????????????????
            // ?? 7. Stick Output - Fixed Decimal Shift + Safety ?????????????????

            // Shift decimal places safely (bring tiny values into 0.0001 ~ 1.0 range)
            // We use floor toward zero style to avoid sign-flipping issues
            float raw_sX = params[0].item<float>();
            float raw_sY = params[1].item<float>();

            // Much softer deadzone + scaling
            const float DEADZONE = 0.03f;        // lower!
            const float SCALE = 2.0f;           // allow overshoot a bit
            //multiply the raws by the sine and cosine of frame count to create a swirling pattern that encourages exploration in all directions of the stick space, while also providing a consistent gradient for learning.
           float sm_sX = raw_sX * SCALE;
           float sm_sY = raw_sY * SCALE;

            //print raw+ :
            printf("Raw stick: (%.8f, %.8f)\n", raw_sX, raw_sY);


            // ?? 8. Execute mem op ?????????????????????????????????????????????
            switch (mem_op) {
            case 0: break;

            case 1: {   // MEM_WRITE
                int slot = mem_addr(params[0].item<float>());
                float situation[MEM_VALS] = {
                    p1_pct, p2_pct, dist_norm, stock_delta * 0.5f + 0.5f,
                    face_norm * 0.5f + 0.5f, 0.f, 0.f, dmg_delta * 0.5f + 0.5f
                };
                float in_buf[INNER_N] = {}, out_buf[INNER_N] = {};
                for (int j = 0; j < MEM_VALS; ++j) in_buf[j] = situation[j];
                run_inner(in_buf, out_buf);
                for (int j = 0; j < MEM_VALS; ++j)
                    gMemSlots[slot][j] = std::clamp(out_buf[j], 0.f, 1.f);
                gMemWritten[slot] = true;
                for (int j = 0; j < 8; ++j) inner_out[j] = out_buf[j];
                break;
            }
            case 2: {   // MEM_READ
                int slot = mem_addr(params[0].item<float>());
                if (gMemWritten[slot]) {
                    float in_buf[INNER_N] = {}, out_buf[INNER_N] = {};
                    for (int j = 0; j < MEM_VALS; ++j) in_buf[j] = gMemSlots[slot][j];
                    run_inner(in_buf, out_buf);
                    for (int j = 0; j < 8; ++j) inner_out[j] = out_buf[j];
                    mem_read_result = gMemSlots[slot][0];
                    c_retrieved = 1.f;
                }
                break;
            }
            case 3: {   // MEM_QUERY
                mem_hit = mem_query(p1_pct, inner_out);
                break;
            }
            case 4: {   // MEM_CONSOLIDATE
                consolidate_inner(12);
                break;
            }
            case 5: {   // MEM_BOOST_WRITE
                int slot = mem_addr(params[0].item<float>());
                float situation[MEM_VALS] = {
                    p1_pct, p2_pct, dist_norm, stock_delta * 0.5f + 0.5f,
                    face_norm * 0.5f + 0.5f, 0.f, 0.f, dmg_delta * 0.5f + 0.5f
                };
                float in_buf[INNER_N] = {}, out_buf[INNER_N] = {};
                for (int j = 0; j < MEM_VALS; ++j) in_buf[j] = situation[j] * 3.f;
                run_inner(in_buf, out_buf);
                for (int j = 0; j < MEM_VALS; ++j)
                    gMemSlots[slot][j] = std::clamp(out_buf[j], 0.f, 1.f);
                gMemWritten[slot] = true;
                for (int j = 0; j < 8; ++j) inner_out[j] = out_buf[j];
                break;
            }
            case 6: {   // MEM_BOOST_QUERY
                float hint_buf[8] = {};
                mem_hit = mem_query(p1_pct, hint_buf);
                if (mem_hit > 0.5f) {
                    action_hint_sX = hint_buf[0] * 2.f - 1.f;
                    action_hint_sY = hint_buf[1] * 2.f - 1.f;
                    action_hint_live = true;
                }
                for (int j = 0; j < 8; ++j) inner_out[j] = hint_buf[j];
                break;
            }
            case 7: {   // RESET_SELF
                std::fill(&gInnerW[0][0], &gInnerW[0][0] + INNER_N * INNER_N, 0.f);
                std::fill(gInnerB, gInnerB + INNER_N, 0.f);
                c_reset = 1.f;
                std::cout << "[NeuraCell] Inner net self-reset (frame " << frame << ")\n";
                break;
            }
            }

            // ?? 9. Passive inner net forward (when no mem op fired) ???????????
            if (mem_op == 0) {
                float in_buf[INNER_N] = {}, out_buf[INNER_N] = {};
                in_buf[0] = p1_pct; in_buf[1] = p2_pct;
                in_buf[2] = dist_norm; in_buf[3] = self_pred_err;
                run_inner(in_buf, out_buf);
                for (int j = 0; j < 8; ++j)
                    inner_out[j] = std::clamp(out_buf[j], 0.f, 1.f);
            }

            // ?? 10. Patch inner net ???????????????????????????????????????????
            apply_patch(patch, PATCH_SCALE);

            // ?? 11. Self-prediction error ?????????????????????????????????????
            {
                float err = 0.f;
                for (int i = 0; i < 8; ++i) {
                    float diff = self_pred[i].item<float>() - state[12 + i];
                    err += diff * diff;
                }
                self_pred_err = self_pred_err * 0.92f + std::sqrt(err / 8.f) * 0.08f;
            }

            // ?? 12. Anti-hang ?????????????????????????????????????????????????
            if (std::abs(sm_sX) < 0.18f && std::abs(sm_sY) < 0.18f &&
                !bA && !bB && !bY && !bZ)
                ++neutral_streak;
            else
                neutral_streak = 0;

            if (neutral_streak > 40) {
                float dir = ((rand() % 2) ? 1.f : -1.f);
               // sm_sX = dir * 0.6f;
               // sm_sY = ((float)rand() / RAND_MAX - 0.5f) * 0.4f;
                neutral_streak = 0;
            }

            // ?? 13. ?? Decode raw params for buffer (before post-processing) ?????
            

            // ?? Write experience to shared buffer ?????????????????????????
            bool is_done = (p1_stks == 0);  // or your stock-lost condition
            WriteExperience(state, bA, bB, bX, bY, bL, bZ, bTrigger,
                mem_op, raw_sX, raw_sY,
                btn_logits, mem_logits, is_done);

            // Update previous percentages for next frame
            prev_p1_pct = p1_pct;
            prev_p2_pct = p2_pct;

            // ?? Send to controller ????????????????????????????????????????
            //bA = bB = bX = bY = bL = bZ = bTrigger = false;
            pad.Send(sm_sX, sm_sY, bA, bB, bX, bY, bL, bZ, bTrigger);
            
            prev_sX = sm_sX;
            prev_sY = sm_sY;

            if (frame > 0 && frame % 600 == 0)
                save_state("neura_cell_sc2.bin");

            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - t0).count();
            Sleep(max(1, 16 - (int)ms));
            ++frame;
        }
    }
    catch (const c10::Error& e) { std::cerr << "\n[Torch] " << e.what() << "\n"; }
    catch (const std::exception& e) { std::cerr << "\n[Error] " << e.what() << "\n"; }

    save_state("neura_cell_sc2.bin");
    return 0;
}