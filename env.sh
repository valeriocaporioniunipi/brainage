BRAINAGE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Base package root. All other relevant folders are relative to this location.
export BRAINAGE_ROOT=$BRAINAGE_DIR
echo "BRAINAGE_ROOT set to " $BRAINAGE_ROOT

# Add the root folder to the $PYTHONPATH so that we can effectively import
# the relevant modules.
export PYTHONPATH=$BRAINAGE_ROOT:$PYTHONPATH
echo "PYTHONPATH set to " $PYTHONPATH
