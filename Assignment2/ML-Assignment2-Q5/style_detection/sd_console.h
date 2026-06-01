#include "console.h"
#include "environment.h"
#include <climits>

using namespace minizero::console;

class SDConsole : public Console {
public:
    SDConsole()
    {
        RegisterFunction("clear_board", this, &SDConsole::CmdClearBoard);
        RegisterFunction("showboard", this, &SDConsole::CmdShowBoard);
        RegisterFunction("play", this, &SDConsole::CmdPlay);
    }

    void CmdClearBoard(const std::vector<std::string>& args)
    {
        if (!checkArgument(args, 0, 0)) { return; }
        env_.reset();
        reply(ConsoleResponse::kSuccess, "");
    }

    void CmdShowBoard(const std::vector<std::string>& args)
    {
        if (!checkArgument(args, 0, 0)) { return; }
        reply(ConsoleResponse::kSuccess, "\n" + env_.toString());
    }

    void CmdPlay(const std::vector<std::string>& args)
    {
        if (!checkArgument(args, 1, INT_MAX)) { return; }
        std::string action_string = args[1];
        if (!env_.act(args)) { return reply(ConsoleResponse::kFail, "Invalid action: \"" + action_string + "\""); }
        reply(ConsoleResponse::kSuccess, "");
    }

private:
    Environment env_;
};