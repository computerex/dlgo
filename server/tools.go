package server

import (
	"fmt"

	"github.com/computerex/dlgo/grammar"
)

// finalizeToolCalls ensures each tool call has proper defaults.
func finalizeToolCalls(toolCalls []ToolCall) []ToolCall {
	for i := range toolCalls {
		if toolCalls[i].ID == "" {
			toolCalls[i].ID = fmt.Sprintf("call_%s", randomHex(8))
		}
		if toolCalls[i].Type == "" {
			toolCalls[i].Type = "function"
		}
	}
	return toolCalls
}

// ToolCallGrammar is a GBNF grammar that constrains output to valid JSON,
// preventing the model from generating malformed JSON like `}}}}]`.
//
// For thinking models, the grammar wraps JSONGrammar with an optional
// <think> block so reasoning models like Qwen3 can still output thinking
// content before the JSON. However, due to prompt-level injection of
// the <think> tag, this wrapper is currently only usable for non-thinking
// models (where plain JSONGrammar is applied).
func ToolCallGrammar(supportsThinking bool) (*grammar.Grammar, error) {
	if supportsThinking {
		// For thinking models, use a grammar that allows optional
		// <think>...</think> before valid JSON.
		src := `root ::= think-block? object ws

think-block ::= "<think>" think-content "</think>" ws
think-content ::= [^<]* ws

object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
array  ::= "[" ws (value ("," ws value)*)? "]" ws
value  ::= object | array | string | number | ("true" | "false" | "null") ws
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws     ::= [ \t\n]*
`
		return grammar.Parse(src)
	}
	return grammar.Parse(grammar.JSONGrammar)
}
