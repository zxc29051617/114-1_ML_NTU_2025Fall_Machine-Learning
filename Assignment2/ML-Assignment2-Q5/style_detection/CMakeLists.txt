add_executable(style main.cpp sd_data_loader.cpp sd_mode_handler.cpp sd_configuration.cpp)
target_link_libraries(
    style
    config
    console
    environment
    learner
    network
    utils
    ${TORCH_LIBRARIES}
)

find_package(pybind11 REQUIRED)
pybind11_add_module(style_py sd_data_loader.cpp pybind.cpp sd_configuration.cpp)
target_link_libraries(
    style_py
    PUBLIC
    config
    environment
    learner
    network
    utils
)