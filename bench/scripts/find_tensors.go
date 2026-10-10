//go:build ignore
package main
import (
    "fmt"
    "os"
    "strings"
    "github.com/computerex/dlgo/format/gguf"
)
func main() {
    gf, _ := gguf.Open(os.Args[1])
    prefix := os.Args[2]
    for i, t := range gf.Tensors {
        if strings.HasPrefix(t.Name, prefix) {
            fmt.Printf("  [%d] %s  shape=%v  type=%d\n", i, t.Name, t.Dimensions, t.Type)
        }
    }
}
