# No global imports here, else tensorflow will (usually) be automatically imported in spawned subprocesses
# which highly slows down the processing time and might lead to conflicts with GPUs

# This allows to use DataParams in spawned sub processes for data preprocessing
