#pragma once

#include "configure_loader.h"

namespace strength_detection {

extern int players_per_batch;
extern int games_per_player;
extern int n_frames;
extern int move_step_to_choose;



void setConfiguration(minizero::config::ConfigureLoader& cl);

} // namespace strength_detection
