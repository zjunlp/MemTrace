"""Global configuration for the LLM memory visualization app."""

# Node color palette by semantic role.
COLOR_USER = "#3B82F6"      # Blue
COLOR_ASSISTANT = "#EF4444"     # Red (legacy alias)
COLOR_SYSTEM = "#F59E0B"    # Yellow

# Optional mapping for data parser / plot components.
COLOR_BY_TYPE = {
    "user": COLOR_USER,
    "assistant": COLOR_ASSISTANT,
    "system": COLOR_SYSTEM,
}

# Layout parameters for the serpentine timeline.
NODES_PER_ROW = 8
HORIZONTAL_SPACING = 180
VERTICAL_SPACING = 180
