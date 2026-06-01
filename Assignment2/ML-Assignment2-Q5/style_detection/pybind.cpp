#include "configuration.h"
#include "sd_configuration.h"
#include "sd_data_loader.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace minizero;
using namespace strength_detection;

std::shared_ptr<Environment> kEnvInstance;

Environment& getEnvInstance()
{
    if (!kEnvInstance) { kEnvInstance = std::make_shared<Environment>(); }
    return *kEnvInstance;
}

PYBIND11_MODULE(style_py, m)
{
    m.def("load_config_file", [](std::string file_name) {
        minizero::env::setUpEnv();
        minizero::config::ConfigureLoader cl;
        strength_detection::setConfiguration(cl);
        cl.loadFromFile(file_name);
        kEnvInstance = std::make_shared<Environment>();
    });
    m.def("load_config_string", [](std::string conf_str) {
        minizero::config::ConfigureLoader cl;
        strength_detection::setConfiguration(cl);
        bool success = cl.loadFromString(conf_str);
        if (success) { kEnvInstance = std::make_shared<Environment>(); }
        return success;
    });
    m.def("use_gumbel", []() { return config::actor_use_gumbel; });
    m.def("get_zero_replay_buffer", []() { return config::zero_replay_buffer; });
    m.def("use_per", []() { return config::learner_use_per; });
    m.def("get_training_step", []() { return config::learner_training_step; });
    m.def("get_training_display_step", []() { return config::learner_training_display_step; });
    m.def("get_batch_size", []() { return config::learner_batch_size; });
    m.def("get_muzero_unrolling_step", []() { return config::learner_muzero_unrolling_step; });
    m.def("get_n_step_return", []() { return config::learner_n_step_return; });
    m.def("get_learning_rate", []() { return config::learner_learning_rate; });
    m.def("get_momentum", []() { return config::learner_momentum; });
    m.def("get_weight_decay", []() { return config::learner_weight_decay; });
    m.def("get_value_loss_scale", []() { return config::learner_value_loss_scale; });
    m.def("get_game_name", []() { return getEnvInstance().name(); });
    m.def("get_nn_file_name", []() { return config::nn_file_name; });
    m.def("get_nn_num_input_channels", []() { return getEnvInstance().getNumInputChannels(); });
    m.def("get_nn_input_channel_height", []() { return getEnvInstance().getInputChannelHeight(); });
    m.def("get_nn_input_channel_width", []() { return getEnvInstance().getInputChannelWidth(); });
    m.def("get_nn_num_hidden_channels", []() { return config::nn_num_hidden_channels; });
    m.def("get_nn_hidden_channel_height", []() { return getEnvInstance().getHiddenChannelHeight(); });
    m.def("get_nn_hidden_channel_width", []() { return getEnvInstance().getHiddenChannelWidth(); });
    m.def("get_nn_num_action_feature_channels", []() { return getEnvInstance().getNumActionFeatureChannels(); });
    m.def("get_nn_num_blocks", []() { return config::nn_num_blocks; });
    m.def("get_nn_action_size", []() { return getEnvInstance().getPolicySize(); });
    m.def("get_nn_num_value_hidden_channels", []() { return config::nn_num_value_hidden_channels; });
    m.def("get_nn_discrete_value_size", []() { return config::nn_discrete_value_size; });
    m.def("get_nn_type_name", []() { return config::nn_type_name; });
    m.def("get_players_per_batch", []() { return strength_detection::players_per_batch; });
    m.def("get_games_per_player", []() { return strength_detection::games_per_player; });
    m.def("get_n_frames", []() { return strength_detection::n_frames; });


    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<std::string>())
        .def("load_data_from_file", &DataLoader::loadDataFromFile)
        .def("Clear_Sgf", &DataLoader::clearDataLoader)
        .def("Check_Sgf", &DataLoader::checkDataLoader)
        .def("get_num_of_player", &DataLoader::getNumOfPlayers)
        .def("get_feature_and_label", [](DataLoader& data_loader, int player_num, int game_id, int start, bool is_train) {
            return py::cast(data_loader.calculateGameFeatures(player_num, game_id, start, is_train));
        })
        .def("get_random_feature_and_label", [](DataLoader& data_loader, int player_num, int game_id, int start, bool is_train) {
            return py::cast(data_loader.calculateRandomMoveGameFeatures(player_num, game_id, start, is_train));
        });
        

    py::class_<SLDataLoader>(m, "SLDataLoader")
        .def(py::init<std::string>())
        .def("initialize", &SLDataLoader::initialize)
        .def("load_data_from_file", &SLDataLoader::loadDataFromFile, py::call_guard<py::gil_scoped_release>())
        .def(
            "sample_data", [](SLDataLoader& data_loader, py::array_t<float>& features, py::array_t<float>& action_features, py::array_t<float>& policy, py::array_t<float>& value, py::array_t<float>& reward, py::array_t<float>& loss_scale, py::array_t<int>& sampled_index) {
                data_loader.getSharedData()->getDataPtr()->features_ = static_cast<float*>(features.request().ptr);
                data_loader.getSharedData()->getDataPtr()->action_features_ = static_cast<float*>(action_features.request().ptr);
                data_loader.getSharedData()->getDataPtr()->policy_ = static_cast<float*>(policy.request().ptr);
                data_loader.getSharedData()->getDataPtr()->value_ = static_cast<float*>(value.request().ptr);
                data_loader.getSharedData()->getDataPtr()->reward_ = static_cast<float*>(reward.request().ptr);
                data_loader.getSharedData()->getDataPtr()->loss_scale_ = static_cast<float*>(loss_scale.request().ptr);
                data_loader.getSharedData()->getDataPtr()->sampled_index_ = static_cast<int*>(sampled_index.request().ptr);
                data_loader.sampleData();
            },
            py::call_guard<py::gil_scoped_release>());

    py::enum_<env::Player>(m, "Player")
        .value("player_none", env::Player::kPlayerNone)
        .value("player_1", env::Player::kPlayer1)
        .value("player_2", env::Player::kPlayer2)
        .value("player_size", env::Player::kPlayerSize)
        .export_values();

    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def(py::init<int, env::Player>())
        .def("next_player", &Action::nextPlayer)
        .def("to_console_string", &Action::toConsoleString)
        .def("get_action_id", &Action::getActionID)
        .def("get_player", &Action::getPlayer);

    py::class_<Environment>(m, "Env")
        .def(py::init<>())
        .def("reset", &Environment::reset)
        .def("act", py::overload_cast<const Action&>(&Environment::act))
        .def("act", py::overload_cast<const std::vector<std::string>&>(&Environment::act))
        .def("get_legal_actions", &Environment::getLegalActions)
        .def("is_legal_action", &Environment::isLegalAction)
        .def("is_terminal", &Environment::isTerminal)
        .def("get_eval_score", &Environment::getEvalScore)
        .def("get_features", [](Environment& env) { return py::cast(env.getFeatures()); })
        .def("get_action_features", &Environment::getActionFeatures)
        .def("to_string", &Environment::toString)
        .def("name", &Environment::name)
        .def("get_turn", &Environment::getTurn)
        .def("get_action_history", &Environment::getActionHistory);

    py::class_<EnvironmentLoader>(m, "EnvLoder")
        .def(py::init<>())
        .def("reset", &EnvironmentLoader::reset)
        .def("load_from_file", &EnvironmentLoader::loadFromFile)
        .def("load_from_string", &EnvironmentLoader::loadFromString)
        .def("load_from_environment", [](EnvironmentLoader& env_loader, const Environment& env) { env_loader.loadFromEnvironment(env); })
        .def("to_string", &EnvironmentLoader::toString)
        .def("get_features", [](EnvironmentLoader& env_loader, const int pos) { return py::cast(env_loader.getFeatures(pos)); });
}
