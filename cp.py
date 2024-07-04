import pstats
from pstats import SortKey

# p = pstats.Stats('restats')
p = pstats.Stats('skill_cprofile')
p.strip_dirs().sort_stats(SortKey.TIME).print_stats()