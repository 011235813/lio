# Cleanup colors
# '@' means "wall"
# 'H' is potential waste spawn point
# 'R' is river cell
# 'S' is stream cell
# 'P' means "player" spawn point
# 'A' means apple spawn point
# 'B' is potential apple spawn point
# ' ' is empty space

CLEANUP_10x10_SYM = [
    '@@@@@@@@@@',
    '@HH   P B@',
    '@RR    BB@',
    '@HH     B@',
    '@RR    BB@',
    '@HH P   B@',
    '@RR    BB@',
    '@HH     B@',
    '@RRP   BB@',
    '@@@@@@@@@@']

# 7x7 map: Agent 0 on river side, Agent 1 on apple side
CLEANUP_SMALL_SYM = [
    '@@@@@@@',
    '@H  PB@',
    '@H   B@',
    '@    B@',
    '@    B@',
    '@ P  B@',
    '@@@@@@@']
