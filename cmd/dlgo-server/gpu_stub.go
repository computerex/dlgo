//go:build !vulkan

package main

import (
	"log"

	"github.com/computerex/dlgo/server"
)

func registerGPU(manager *server.ModelManager) {
	log.Println("GPU support not compiled in (build with -tags vulkan)")
}
