#pragma once

#include "data_loader.h"
#include "environment.h"
#include "sd_configuration.h"
#include "sgf_loader.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

class DataLoader {
public:
    DataLoader(std::string conf_file);

    void loadDataFromFile(const std::string& file_name);

    std::vector<float> calculateGameFeatures(int player_num, int game_id, int start, bool is_train);
    std::vector<float> calculateRandomMoveGameFeatures(int player_num, int game_id, int start, bool is_train);
    void clearDataLoader();
    void checkDataLoader();
    int getNumOfPlayers();
    static inline int getMoveStepToChoose() { return strength_detection::move_step_to_choose; }
    static inline int getGamesPerPlayer() { return strength_detection::games_per_player; }
    static inline int getNFrames() { return strength_detection::n_frames; }

private:
    Environment env_;

    std::map<std::string, std::vector<EnvironmentLoader>> env_loaders_;
};

class SLDataLoaderThread : public minizero::learner::DataLoaderThread {
public:
    SLDataLoaderThread(int id, std::shared_ptr<minizero::utils::BaseSharedData> shared_data)
        : DataLoaderThread(id, shared_data) {}

protected:
    bool addEnvironmentLoader() override;
};

class SLDataLoader : public minizero::learner::DataLoader {
public:
    SLDataLoader(const std::string& conf_file_name);

    std::shared_ptr<minizero::utils::BaseSlaveThread> newSlaveThread(int id) override { return std::make_shared<SLDataLoaderThread>(id, shared_data_); }
};
