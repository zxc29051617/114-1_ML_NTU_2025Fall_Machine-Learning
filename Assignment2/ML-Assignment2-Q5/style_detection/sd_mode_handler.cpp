#include "sd_mode_handler.h"
#include "create_network.h"
#include "random.h"
#include <memory>
#include <vector>

namespace strength_detection {

using namespace minizero;

SDModeHandler::SDModeHandler()
{
    RegisterFunction("sgf_policy", this, &SDModeHandler::runSGFPolicy);
}

void SDModeHandler::runSGFPolicy()
{
    std::shared_ptr<network::Network> network = network::createNetwork(config::nn_file_name, 3);
    std::shared_ptr<network::AlphaZeroNetwork> az_network = std::static_pointer_cast<network::AlphaZeroNetwork>(network);
    std::vector<EnvironmentLoader> env_loaders = loadEnvironmentLoaders();

    for (auto& env_loader : env_loaders) {
        Environment env;
        for (auto& action_pair : env_loader.getActionPairs()) {
            az_network->pushBack(env.getFeatures());
            env.act(action_pair.first);
        }

        std::vector<std::shared_ptr<network::NetworkOutput>> network_outputs = az_network->forward();
        for (auto& output : network_outputs) {
            std::shared_ptr<network::AlphaZeroNetworkOutput> az_output = std::static_pointer_cast<network::AlphaZeroNetworkOutput>(output);
            for (auto& p : az_output->policy_) { std::cerr << p << " "; }
            std::cerr << std::endl;
        }
    }
}

std::vector<EnvironmentLoader> SDModeHandler::loadEnvironmentLoaders()
{
    std::vector<EnvironmentLoader> env_loaders;

    // TODO: read sgf

    return env_loaders;
}

} // namespace strength_detection
