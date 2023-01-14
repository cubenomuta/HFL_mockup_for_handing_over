docker run -it --rm\
  -u 1000:1000 \
  --name test \
  --mount type=bind,source="$(pwd)"/shell/,target=/home/"${USER}"/project/shell/,readonly \
  --mount type=bind,source="$(pwd)"/data/,target=/home/"${USER}"/project/data/ \
  --mount type=bind,source="$(pwd)"/local/,target=/home/"${USER}"/project/local/,readonly \
  --mount type=bind,source="$(pwd)"/src/,target=/home/"${USER}"/project/src/,readonly \
  --env CID="$1" \
  flower_mockup_pi4:latest