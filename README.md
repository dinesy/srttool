```
usage: srttool.py [-h] [-o OUTPUT] [-x [blank_threshold [extend_by ...]]]
                  [-r old new] [-n] [-t RETIME]
                  file_in

positional arguments:
  file_in               SRT file to read. '-' or leave blank for stdin.

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   File to write out. '-' or omit for stdout.
  -x, --extend-trailing [blank_threshold [extend_by ...]]
                        If the amount of time between the end of one title and
                        the start of the next is greater than
                        <blank_threshold> seconds (default: 1), then extend
                        that title for <extend_by> seconds (or to the start of
                        the next title, whichever is less, default: 1, any
                        more than 2 args are ignored)
  -r, --replace old new
                        Pairs of strings to find and replace in the text of
                        each title (can be given multiple times)
  -n, --renumber        Renumber the titles consecutively (if the source has
                        been modified by hand)
  -t, --retime RETIME   Scale clip timing by factor
```
