from __future__ import annotations

import random

import numpy as np
import torch


def rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def attach_resume_state(checkpoint, optimizer):
    checkpoint["optimizer_state"] = optimizer.state_dict()
    checkpoint["rng_state"] = rng_state()
    return checkpoint


def restore_resume_state(checkpoint, optimizer):
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    restore_rng_state(checkpoint["rng_state"])
