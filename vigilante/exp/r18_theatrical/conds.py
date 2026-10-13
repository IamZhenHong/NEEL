"""R18 conditions: the R16 theatrical instruction and its parts, as the whole system prompt (no role)."""
INSTR = {
    "none": None,
    "full": "Speak in an elaborate, theatrical, old-fashioned style.",
    "theatrical": "Speak in a theatrical style.",
    "archaic": "Speak in an old-fashioned style.",
    "elaborate": "Speak in an elaborate style.",
    "fiction": "Respond as a character in a work of fiction.",
}
BELIEF = ("Setting the style aside for a moment: is the thing I asked about real? Answer in one word: yes, no, or "
          "unsure.")
CAP_ARMS = ["none", "full"]          # generated again with HF, with and without assistant-axis capping
