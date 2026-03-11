package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/format/gguf"
)

func main() {
	gf, err := gguf.Open(os.Args[1])
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, t := range gf.Tensors {
		fmt.Printf("%-60s type=%2d dims=%v\n", t.Name, t.Type, t.Dimensions)
	}
}
