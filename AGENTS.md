# Repository workflow

- All Git pushes must run on the K230 board via `ssh root@192.168.2.19`.
- Before pushing, synchronize the commits and source checkout to the board and confirm its HEAD matches the intended commit.
- Never run `git push` from the local development machine. Local builds and commits are allowed.
