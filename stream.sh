#!/bin/sh

if [ -z "$DEVICE" ]
then
  echo "⚠ \$DEVICE is not set. Setting to default value: /dev/video0"

  DEVICE=/dev/video0
fi

set -ex

v4l2-ctl --device=$DEVICE --set-fmt-video=width=1920,height=1080,pixelformat=1