package main
import (
    "fmt"
    "os"
    "github.com/computerex/dlgo/format/gguf"
)
func main() {
    m, _ := gguf.Open(os.Args[1])
    for k, v := range m.Metadata {
        fmt.Printf("  %s = %v\n", k, v)
    }
}
