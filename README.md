# NuAngel

NeuraCell SC2 — A real-time reinforcement learning agent for Soul Calibur II (Dolphin) using a custom LSTM + memory-augmented neural network.

Built from scratch with:
- TorchScript model running at 60 FPS in a C++ hook (ViGEm + Dolphin)
- Independent Bernoulli button heads (7 buttons) + 3-bit memory operation head
- Tactical working memory system (16 slots) with read/write/query/consolidate operations
- Online PPO training with shared memory buffer (mmap)
- Inner adaptive network with live patching

Architecture Overview
NeuraCellNet consists of two tightly coupled networks:

Outer Network (Policy + Value)
A 2-layer LSTM that processes the last 64 frames of game state. It outputs:
7 independent button logits (A, B, X, Y, L, Z, Shoulder) using Bernoulli heads — allowing natural multi-button combos
3-bit memory operation logits (MEM_WRITE, MEM_READ, MEM_QUERY, CONSOLIDATE, etc.)
Continuous stick parameters (X, Y)
Inner network patch updates
Self-prediction target and value estimate

Inner Network (32×32 Leaky ReLU)
A small, fast, fully connected "working memory" network that runs entirely on the CPU every frame.
Its purpose is to provide rapid, low-level tactical processing and act as a dynamic knowledge store.
The outer network can:
Write important situations into the inner net via MEM_WRITE
Read and process stored patterns via MEM_READ
Query for similar situations via MEM_QUERY
Consolidate knowledge or reset itself
This dual-network design allows the agent to combine fast reactive control (inner net) with longer-term strategic reasoning (LSTM + memory ops).

Key Features

Real-time inference hook using TorchScript + ViGEm (Xbox 360 virtual controller)
Shared memory replay buffer for online PPO training
Tactical working memory (16 slots × 8 values) with consolidation
Frame-phase awareness via sin/cos channels
Fully independent button heads for true combo freedom

To Run:
Launch train3.py and wait for it to generate the necessary files then launch SC2Hook.exe to run in parallel.

Issues: Several buttons pressing at once
        Reward logic zeroed out in trainer3.py for debugging reasons (didnt help)

Basically, not functioning but a good start. As-is. If anyone wants to nudge it in the right direction feel free to do so (let me know)!
  What youll need:

  libtorch-win-shared-with-deps-2.11.0+cu126
  https://github.com/nefarius/ViGEmClient
  Dolphin 4.0-8065 (Earlier versions had fixed memory addresses, easier to hook into)
    
  Not sure if im allowed to redestribute the .dlls will add later if allowed.
