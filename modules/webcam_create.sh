#!/bin/bash
set -e

rmmod v4l2loopback 2>/dev/null || true
modprobe v4l2loopback "$@"
