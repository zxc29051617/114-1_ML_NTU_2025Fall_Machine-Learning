#include "sd_mode_handler.h"

int main(int argc, char* argv[])
{
    strength_detection::SDModeHandler mode_handler;
    mode_handler.run(argc, argv);
    return 0;
}
