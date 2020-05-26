from chainermnx.optimizers.optimizers import create_multi_node_optimizer
from chainermnx.optimizers.spatial_optimizer import create_spatial_optimizer
from chainermnx.optimizers.hybrid_optimizer import create_hybrid_multi_node_optimizer

# We created this to avoid loss problem. In short bypass calculating loss.
# Nguyen: I comment this line because Albert deleted the file
#from chainermnx.optimizers.hybrid_optimizer_alpha import create_hybrid_multi_node_optimizer_alpha  # No idea why I created this








