#include "sd_data_loader.h"
#include "configuration.h"
#include "environment.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

DataLoader::DataLoader(std::string conf_file)
{
    minizero::config::ConfigureLoader cl;
    strength_detection::setConfiguration(cl);
    minizero::config::setConfiguration(cl);
    cl.loadFromFile(conf_file);
    minizero::env::setUpEnv();
    env_loaders_.clear();
}

void DataLoader::loadDataFromFile(const std::string& file_name)
{
    std::cerr << "Read " << file_name << "..." << std::endl;
    std::ifstream fin(file_name, std::ifstream::in);

    std::string player_name;
    std::getline(fin, player_name); // The first line in Sgf

    for (std::string content; std::getline(fin, content);) {
        minizero::utils::SGFLoader sgf_loader;
        if (!sgf_loader.loadFromString(content)) { continue; }
        if (std::stoi(sgf_loader.getTags().at("SZ")) != 19) { continue; }

        EnvironmentLoader env_loader;
        env_loader.reset();
        env_loader.addTag("SZ", sgf_loader.getTags().at("SZ"));
        env_loader.addTag("KM", sgf_loader.getTags().at("KM"));
        env_loader.addTag("RE", std::to_string(sgf_loader.getTags().at("RE")[0] == 'B' ? 1.0f : -1.0f));
        env_loader.addTag("PB", sgf_loader.getTags().at("PB"));
        env_loader.addTag("PW", sgf_loader.getTags().at("PW"));

        for (auto& action_string : sgf_loader.getActions()) {
            env_loader.addActionPair(Action(action_string.first, std::stoi(sgf_loader.getTags().at("SZ"))), action_string.second);
        }

        std::string PB_name = env_loader.getTag("PB");
        std::string PW_name = env_loader.getTag("PW");

        if (player_name == PB_name) {
            env_loaders_[PB_name].push_back(env_loader);
        } else if (player_name == PW_name) {
            env_loaders_[PW_name].push_back(env_loader);
        }
    }
}

std::vector<float> DataLoader::calculateGameFeatures(int player_num, int game_id, int start, bool is_train)
{
    std::vector<float> features_;
    std::vector<float> game_features_;
    unsigned int local_seed = std::time(NULL);
    game_features_.clear();
    int n_frames = getNFrames();
    int move_step_to_choose = getMoveStepToChoose();
    n_frames *= move_step_to_choose;
    auto it = env_loaders_.begin();
    std::advance(it, player_num);
    std::string player = it->first;
    int start_cal = start;
    int games_per_player = 1;
    if (is_train) games_per_player = getGamesPerPlayer();
    for (int k = 0; k < games_per_player; k++) {
        int choose_move = 0;
        env_.reset();

        if (is_train) { // train mode
            game_id = rand_r(&local_seed) % env_loaders_[player].size();
            int boundary = env_loaders_[player][game_id].getActionPairs().size() - n_frames;
            if (start > boundary) boundary = start;
            start_cal = rand_r(&local_seed) % (boundary - start + 1) + start;
        }

        int move_counts = env_loaders_[player][game_id].getActionPairs().size();
        std::string PB_name = env_loaders_[player][game_id].getTag("PB");
        minizero::env::Player color = PB_name == player ? minizero::env::Player::kPlayer1 : minizero::env::Player::kPlayer2;
        int step = 0;
        for (int i = 0; i < move_counts; ++i) {
            const Action& action = env_loaders_[player][game_id].getActionPairs()[i].first;
            env_.act(action);
            if (i >= start_cal) {
                if (action.getPlayer() == color) {
                    if (step < move_step_to_choose / 2 - 1) {
                        step++;
                        continue;
                    }
                    choose_move++;
                    features_ = env_.getFeatures(); // (1,18*19*19)
                    for (size_t j = 0; j < features_.size(); j++) {
                        game_features_.push_back(features_[j]); // (n_frames,18*19*19)
                    }
                    step = 0;
                }
            }
            if (choose_move >= n_frames / move_step_to_choose) break;
        }

        if (choose_move < n_frames / move_step_to_choose) {
            for (int i = 0; i < (n_frames / move_step_to_choose - choose_move); i++) {
                for (size_t j = 0; j < features_.size(); j++) {
                    game_features_.push_back(0.0f);
                }
            }
        }
    }
    return game_features_;
}
std::vector<float> DataLoader::calculateRandomMoveGameFeatures(int player_num, int game_id, int start, bool is_train)
{
    std::vector<float> features_;
    std::vector<float> game_features_;
    unsigned int local_seed = std::time(NULL);
    game_features_.clear();
    int n_frames = getNFrames();

    int games_per_player = 1;
    if (is_train) games_per_player = getGamesPerPlayer();

    auto it = env_loaders_.begin();
    std::advance(it, player_num);
    std::string player = it->first;

    for (int k = 0; k < games_per_player; k++) {
        int choose_move = 0;
        env_.reset();

        if (is_train) game_id = rand_r(&local_seed) % env_loaders_[player].size();

        int move_counts = env_loaders_[player][game_id].getActionPairs().size();
        std::string PB_name = env_loaders_[player][game_id].getTag("PB");
        minizero::env::Player color = PB_name == player ? minizero::env::Player::kPlayer1 : minizero::env::Player::kPlayer2;
        std::vector<int> random_move;
        for (int i = 0; i < move_counts; i++) {
            const Action& action = env_loaders_[player][game_id].getActionPairs()[i].first;
            if (action.getPlayer() == color)
                random_move.push_back(i);
        }
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::srand(seed);
        std::random_shuffle(random_move.begin(), random_move.end());

        std::sort(random_move.begin(), random_move.begin() + n_frames);

        int temp = 0;
        for (int i = 0; i < move_counts; ++i) {
            const Action& action = env_loaders_[player][game_id].getActionPairs()[i].first;
            env_.act(action);
            features_ = env_.getFeatures(); // (1,18*19*19)

            if (i == random_move[temp]) {
                choose_move++;
                temp++;
                for (size_t j = 0; j < features_.size(); j++) {
                    game_features_.push_back(features_[j]); // (n_frames,18*19*19)
                }
            }
            if (choose_move >= n_frames) break;
        }

        if (choose_move < n_frames) {
            for (int i = 0; i < (n_frames - choose_move); i++) {
                for (size_t j = 0; j < features_.size(); j++) {
                    game_features_.push_back(0.0f);
                }
            }
        }
    }
    return game_features_;
}
int DataLoader::getNumOfPlayers()
{
    return env_loaders_.size();
}
void DataLoader::clearDataLoader()
{
    for (auto it = env_loaders_.begin(); it != env_loaders_.end(); ++it) {
        env_loaders_[it->first].clear();
    }
}

void DataLoader::checkDataLoader()
{
    for (auto it = env_loaders_.begin(); it != env_loaders_.end(); ++it) {
        std::cout << "player:" << it->first << ",size=" << env_loaders_[it->first].size() << std::endl;
    }
}

SLDataLoader::SLDataLoader(const std::string& conf_file_name)
    : minizero::learner::DataLoader("")
{
    minizero::env::setUpEnv();
    minizero::config::ConfigureLoader cl;
    strength_detection::setConfiguration(cl);
    cl.loadFromFile(conf_file_name);
}

bool SLDataLoaderThread::addEnvironmentLoader()
{
    std::string env_string = getSharedData()->getNextEnvString();
    if (env_string.empty()) { return false; }
    if (env_string.find("(") == std::string::npos) { return true; }

    minizero::utils::SGFLoader sgf_loader;
    if (!sgf_loader.loadFromString(env_string)) { return true; }
    if (std::stoi(sgf_loader.getTags().at("SZ")) != 19) { return true; }

    EnvironmentLoader env_loader;
    env_loader.reset();
    env_loader.addTag("SZ", sgf_loader.getTags().at("SZ"));
    env_loader.addTag("KM", sgf_loader.getTags().at("KM"));
    env_loader.addTag("RE", std::to_string(sgf_loader.getTags().at("RE")[0] == 'B' ? 1.0f : -1.0f));
    env_loader.addTag("PB", sgf_loader.getTags().at("PB"));
    env_loader.addTag("PW", sgf_loader.getTags().at("PW"));

    for (auto& action_string : sgf_loader.getActions()) { env_loader.addActionPair(Action(action_string.first, std::stoi(sgf_loader.getTags().at("SZ"))), action_string.second); }
    getSharedData()->replay_buffer_.addData(env_loader);
    return true;
}
