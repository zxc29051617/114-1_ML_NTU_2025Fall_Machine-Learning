#pragma once

#include "environment.h"
#include "mode_handler.h"
#include "sd_configuration.h"
#include <vector>

namespace strength_detection {

class SDModeHandler : public minizero::console::ModeHandler {
public:
    SDModeHandler();

protected:
    void setDefaultConfiguration(minizero::config::ConfigureLoader& cl) override { strength_detection::setConfiguration(cl); }
    void runSGFPolicy();

    std::vector<EnvironmentLoader> loadEnvironmentLoaders();
};

} // namespace strength_detection
