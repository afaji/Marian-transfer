test -f $MRT_MARIAN/marian-server || exit 1
test -f $MRT_MODELS/wmt16_systems/en-de/model.npz || exit 1
python -c "import websocket"
