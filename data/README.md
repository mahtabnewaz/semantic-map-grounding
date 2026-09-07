# Data

Canonical layout (CLAUDE.md):

- `houseexpo/` — floor plans from the HouseExpo dataset (IROS 2020)
- `real_maps/` — PGM+YAML occupancy grids exported from the robot
- `platforms/` — per-platform vendor docs, URDFs, package.xml, message
  defs (inputs to the onboarding study)

## Currently present (as downloaded — not yet relocated)

- `HouseExpo_png/png/` — HouseExpo floor-plan PNGs. Belongs under
  `houseexpo/`.
- `real_my_room_simple_map/` — a real `map_01.pgm` + `map_01.yaml`
  export. Belongs under `real_maps/`.

These were left where they were downloaded so nothing breaks. Move them
into the canonical folders above when you're ready (say the word and I'll
do it).

Large/raw data should stay out of version control — see `.gitignore`.
