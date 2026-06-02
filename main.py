"""
main.py
-------
Entry point for the Hand Gesture Game project.

Presents a simple CLI menu to launch either the Hand HUD visualizer
or the Snake Game.

Usage:
    python main.py
"""

import sys


def print_banner():
    banner = r"""
  _   _                 _    ____           _
 | | | | __ _ _ __   __| |  / ___| ___  ___| |_ _   _ _ __ ___
 | |_| |/ _` | '_ \ / _` | | |  _ / _ \/ __| __| | | | '__/ _ \
 |  _  | (_| | | | | (_| | | |_| |  __/\__ \ |_| |_| | | |  __/
 |_| |_|\__,_|_| |_|\__,_|  \____|\___||___/\__|\__,_|_|  \___|

          Snake Game + Hand HUD  |  Powered by MediaPipe
    """
    print(banner)


def menu():
    print_banner()
    print("  [1]  Hand HUD Visualizer  — live hand tracking overlay")
    print("  [2]  Snake Game           — control snake with your hand")
    print("  [q]  Quit")
    print()

    while True:
        choice = input("  Select option: ").strip().lower()
        if choice == "1":
            from hand_hud import run
            run()
            break
        elif choice == "2":
            from snake_game import run_game
            run_game()
            break
        elif choice in ("q", "quit", "exit"):
            print("Bye!")
            sys.exit(0)
        else:
            print("  Invalid choice. Enter 1, 2, or q.")


if __name__ == "__main__":
    menu()
