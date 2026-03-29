// Package grammar implements GBNF grammar-constrained decoding for LLM inference.
// It is a Go port of llama.cpp's grammar system, enabling structured output
// (e.g., valid JSON) by masking invalid tokens during sampling.
package grammar

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// ElementType represents the type of a grammar element.
type ElementType int

const (
	TypeEnd         ElementType = iota // end of rule definition
	TypeAlt                            // start of alternate definition for rule
	TypeRuleRef                        // non-terminal: reference to rule
	TypeChar                           // terminal: character (code point)
	TypeCharNot                        // inverse char(s) [^a], [^a-b]
	TypeCharRngUp                      // upper bound of inclusive range [a-z]
	TypeCharAlt                        // alternate char in class [ab]
	TypeCharAny                        // any character (.)
)

// Element is a single element in a grammar rule.
type Element struct {
	Type  ElementType
	Value uint32 // Unicode code point or rule ID
}

// Rule is a sequence of elements (may contain ALT separators and END terminator).
type Rule []Element

// Stack is a parse position stack: pointers into rule elements.
// We represent pointers as (ruleID, elementIndex) pairs for safety in Go.
type StackPos struct {
	Rule    int
	Element int
}

type Stack []StackPos

// Grammar holds the parsed rules and live parse state.
type Grammar struct {
	Rules  []Rule  // static rules (indexed by rule ID)
	Stacks []Stack // live parse position stacks

	// partial UTF-8 state
	PartialValue   uint32
	PartialNRemain int
}

// isEndOfSequence returns true if the element at pos is END or ALT.
func isEndOfSequence(elem Element) bool {
	return elem.Type == TypeEnd || elem.Type == TypeAlt
}

// getElement safely retrieves an element from the grammar rules.
func (g *Grammar) getElement(pos StackPos) Element {
	return g.Rules[pos.Rule][pos.Element]
}

// next returns the position of the next element in the same rule.
func next(pos StackPos) StackPos {
	return StackPos{pos.Rule, pos.Element + 1}
}

// stacksEqual checks if two stacks are identical.
func stacksEqual(a, b Stack) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// containsStack checks if the stacks slice already contains this stack.
func containsStack(stacks []Stack, s Stack) bool {
	for _, existing := range stacks {
		if stacksEqual(existing, s) {
			return true
		}
	}
	return false
}

// advanceStack transforms a grammar pushdown stack into N possible stacks,
// all ending at a terminal element (char range or end).
func (g *Grammar) advanceStack(stack Stack, newStacks *[]Stack) {
	if len(stack) == 0 {
		if !containsStack(*newStacks, stack) {
			s := make(Stack, 0)
			*newStacks = append(*newStacks, s)
		}
		return
	}

	pos := stack[len(stack)-1]
	elem := g.getElement(pos)

	switch elem.Type {
	case TypeRuleRef:
		ruleID := int(elem.Value)
		rule := g.Rules[ruleID]
		// iterate over alternatives of the referenced rule
		subIdx := 0
		for {
			// build new stack without the top (current RULE_REF)
			newStack := make(Stack, len(stack)-1)
			copy(newStack, stack[:len(stack)-1])

			// if elements follow the rule ref, push as return point
			nextPos := next(pos)
			if nextPos.Element < len(g.Rules[pos.Rule]) && !isEndOfSequence(g.getElement(nextPos)) {
				newStack = append(newStack, nextPos)
			}

			// if this alternative is non-empty, push its first element
			if subIdx < len(rule) && !isEndOfSequence(rule[subIdx]) {
				newStack = append(newStack, StackPos{ruleID, subIdx})
			}

			g.advanceStack(newStack, newStacks)

			// scan to end of this alternative
			for subIdx < len(rule) && !isEndOfSequence(rule[subIdx]) {
				subIdx++
			}
			if subIdx < len(rule) && rule[subIdx].Type == TypeAlt {
				subIdx++ // move to start of next alternative
			} else {
				break
			}
		}

	case TypeChar, TypeCharNot, TypeCharAny:
		if !containsStack(*newStacks, stack) {
			s := make(Stack, len(stack))
			copy(s, stack)
			*newStacks = append(*newStacks, s)
		}

	default:
		// TypeEnd, TypeAlt, TypeCharAlt, TypeCharRngUp should not be on top of stack
		panic(fmt.Sprintf("grammar: unexpected element type %d on stack top", elem.Type))
	}
}

// matchChar checks whether chr satisfies the char range at the given position.
// Returns (matched, nextPos after the char range elements).
func (g *Grammar) matchChar(pos StackPos, chr uint32) (bool, StackPos) {
	elem := g.getElement(pos)
	found := false
	isPositive := elem.Type == TypeChar || elem.Type == TypeCharAny

	cur := pos
	for {
		e := g.getElement(cur)
		nextE := cur
		nextE.Element++

		if nextE.Element < len(g.Rules[cur.Rule]) && g.Rules[cur.Rule][nextE.Element].Type == TypeCharRngUp {
			// inclusive range [a-z]
			upper := g.Rules[cur.Rule][nextE.Element].Value
			if e.Value <= chr && chr <= upper {
				found = true
			}
			cur.Element += 2
		} else if e.Type == TypeCharAny {
			found = true
			cur.Element++
		} else {
			// exact char match
			if e.Value == chr {
				found = true
			}
			cur.Element++
		}

		// check if next element continues the char class
		if cur.Element < len(g.Rules[cur.Rule]) && g.Rules[cur.Rule][cur.Element].Type == TypeCharAlt {
			continue
		}
		break
	}

	return found == isPositive, cur
}

// matchPartialChar checks if a partial UTF-8 sequence could satisfy the char range.
func (g *Grammar) matchPartialChar(pos StackPos, partialValue uint32, nRemain int) bool {
	elem := g.getElement(pos)
	isPositive := elem.Type == TypeChar || elem.Type == TypeCharAny

	if nRemain < 0 || (nRemain == 1 && partialValue < 2) {
		return false
	}

	low := partialValue << (uint(nRemain) * 6)
	high := low | ((1 << (uint(nRemain) * 6)) - 1)

	if low == 0 {
		if nRemain == 2 {
			low = 1 << 11
		} else if nRemain == 3 {
			low = 1 << 16
		}
	}

	cur := pos
	for {
		e := g.getElement(cur)
		nextE := cur
		nextE.Element++

		if nextE.Element < len(g.Rules[cur.Rule]) && g.Rules[cur.Rule][nextE.Element].Type == TypeCharRngUp {
			upper := g.Rules[cur.Rule][nextE.Element].Value
			if e.Value <= high && low <= upper {
				return isPositive
			}
			cur.Element += 2
		} else if e.Type == TypeCharAny {
			return true
		} else {
			if low <= e.Value && e.Value <= high {
				return isPositive
			}
			cur.Element++
		}

		if cur.Element < len(g.Rules[cur.Rule]) && g.Rules[cur.Rule][cur.Element].Type == TypeCharAlt {
			continue
		}
		break
	}

	return !isPositive
}

// candidate represents a token being checked against the grammar.
type candidate struct {
	index      int       // index in the logits/token array
	codePoints []uint32  // remaining UTF-8 code points (0-terminated)
	cpOffset   int       // current offset into codePoints
	partialVal uint32    // partial UTF-8 value
	partialN   int       // partial UTF-8 remaining bytes
}

// rejectCandidatesForStack returns candidates rejected by a single stack.
func (g *Grammar) rejectCandidatesForStack(stack Stack, candidates []candidate) []candidate {
	var rejects []candidate

	if len(stack) == 0 {
		// grammar complete: reject tokens that have remaining content
		for _, tok := range candidates {
			if tok.cpOffset < len(tok.codePoints) && tok.codePoints[tok.cpOffset] != 0 {
				rejects = append(rejects, tok)
			} else if tok.partialN != 0 {
				rejects = append(rejects, tok)
			}
		}
		return rejects
	}

	stackPos := stack[len(stack)-1]
	elem := g.getElement(stackPos)
	_ = elem

	var nextCandidates []candidate

	for _, tok := range candidates {
		if tok.cpOffset >= len(tok.codePoints) || tok.codePoints[tok.cpOffset] == 0 {
			// end of token's codepoints
			if tok.partialN != 0 && !g.matchPartialChar(stackPos, tok.partialVal, tok.partialN) {
				rejects = append(rejects, tok)
			}
		} else {
			matched, _ := g.matchChar(stackPos, tok.codePoints[tok.cpOffset])
			if matched {
				advanced := tok
				advanced.cpOffset++
				nextCandidates = append(nextCandidates, advanced)
			} else {
				rejects = append(rejects, tok)
			}
		}
	}

	if len(nextCandidates) == 0 {
		return rejects
	}

	// advance stack position past the char match
	_, afterPos := g.matchChar(stackPos, 0) // get position after char elements

	stackAfter := make(Stack, len(stack)-1)
	copy(stackAfter, stack[:len(stack)-1])
	if afterPos.Element < len(g.Rules[afterPos.Rule]) && !isEndOfSequence(g.getElement(afterPos)) {
		stackAfter = append(stackAfter, afterPos)
	}

	var nextStacks []Stack
	g.advanceStack(stackAfter, &nextStacks)

	nextRejects := g.rejectCandidates(nextStacks, nextCandidates)
	for _, rej := range nextRejects {
		rej.cpOffset-- // restore original offset
		rejects = append(rejects, rej)
	}

	return rejects
}

// rejectCandidates returns candidates rejected by ALL stacks (intersection of rejections).
func (g *Grammar) rejectCandidates(stacks []Stack, candidates []candidate) []candidate {
	if len(stacks) == 0 || len(candidates) == 0 {
		return candidates // reject everything if no stacks
	}

	rejects := g.rejectCandidatesForStack(stacks[0], candidates)
	for i := 1; i < len(stacks); i++ {
		rejects = g.rejectCandidatesForStack(stacks[i], rejects)
	}
	return rejects
}

// AcceptChar accepts a single Unicode code point, advancing the grammar state.
func (g *Grammar) AcceptChar(chr uint32) {
	var newStacks []Stack

	for _, stack := range g.Stacks {
		if len(stack) == 0 {
			continue
		}
		pos := stack[len(stack)-1]
		matched, afterPos := g.matchChar(pos, chr)
		if matched {
			newStack := make(Stack, len(stack)-1)
			copy(newStack, stack[:len(stack)-1])
			if afterPos.Element < len(g.Rules[afterPos.Rule]) && !isEndOfSequence(g.getElement(afterPos)) {
				newStack = append(newStack, afterPos)
			}
			g.advanceStack(newStack, &newStacks)
		}
	}

	g.Stacks = newStacks
}

// AcceptToken accepts a token string, advancing the grammar state through each character.
func (g *Grammar) AcceptToken(piece string) {
	codePoints, partial := decodeUTF8(piece, g.PartialValue, g.PartialNRemain)

	// Process each codepoint (skip the terminating 0)
	for i := 0; i < len(codePoints)-1; i++ {
		g.AcceptChar(codePoints[i])
	}

	g.PartialValue = partial.value
	g.PartialNRemain = partial.nRemain
}

// IsComplete returns true if the grammar has reached a valid end state.
func (g *Grammar) IsComplete() bool {
	for _, stack := range g.Stacks {
		if len(stack) == 0 {
			return true
		}
	}
	return false
}

// ApplyToLogits masks logits for tokens that don't fit the grammar.
// tokenPieces maps token ID -> decoded string for each token in the vocabulary.
// Tokens that violate the grammar get logit = -Inf.
func (g *Grammar) ApplyToLogits(logits []float32, tokenPieces []string, eosTokens map[int32]bool) {
	if len(g.Stacks) == 0 {
		return
	}

	allowEOS := g.IsComplete()

	var candidates []candidate

	for i := 0; i < len(logits); i++ {
		if logits[i] == float32(math.Inf(-1)) {
			continue // already masked
		}

		// Handle EOS tokens
		if eosTokens[int32(i)] {
			if !allowEOS {
				logits[i] = float32(math.Inf(-1))
			}
			continue
		}

		piece := tokenPieces[i]
		if len(piece) == 0 {
			logits[i] = float32(math.Inf(-1))
			continue
		}

		codePoints, partial := decodeUTF8(piece, g.PartialValue, g.PartialNRemain)
		candidates = append(candidates, candidate{
			index:      i,
			codePoints: codePoints,
			cpOffset:   0,
			partialVal: partial.value,
			partialN:   partial.nRemain,
		})
	}

	rejects := g.rejectCandidates(g.Stacks, candidates)
	for _, rej := range rejects {
		logits[rej.index] = float32(math.Inf(-1))
	}
}

// ========== UTF-8 helpers ==========

type partialUTF8 struct {
	value  uint32
	nRemain int
}

func decodeUTF8(src string, partialValue uint32, partialNRemain int) ([]uint32, partialUTF8) {
	lookup := [16]int{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4}

	var codePoints []uint32
	pos := 0
	value := partialValue
	nRemain := partialNRemain

	bytes := []byte(src)

	// continue previous partial decode
	for pos < len(bytes) && nRemain > 0 {
		b := bytes[pos]
		if (b >> 6) != 2 {
			// invalid continuation byte
			codePoints = append(codePoints, 0)
			return codePoints, partialUTF8{0, -1}
		}
		value = (value << 6) + uint32(b&0x3F)
		pos++
		nRemain--
	}

	if partialNRemain > 0 && nRemain == 0 {
		codePoints = append(codePoints, value)
	}

	// decode remaining bytes
	for pos < len(bytes) {
		firstByte := bytes[pos]
		highbits := firstByte >> 4
		nRemain = lookup[highbits] - 1

		if nRemain < 0 {
			// invalid byte
			codePoints = append(codePoints, 0)
			return codePoints, partialUTF8{0, nRemain}
		}

		mask := byte((1 << (7 - nRemain)) - 1)
		value = uint32(firstByte & mask)
		pos++

		for pos < len(bytes) && nRemain > 0 {
			value = (value << 6) + uint32(bytes[pos]&0x3F)
			pos++
			nRemain--
		}
		if nRemain == 0 {
			codePoints = append(codePoints, value)
		}
	}

	codePoints = append(codePoints, 0) // terminating zero
	return codePoints, partialUTF8{value, nRemain}
}

// ========== GBNF Parser ==========

// Parser parses GBNF grammar text into rules.
type Parser struct {
	SymbolIDs map[string]int
	Rules     []Rule
	nextGenID int
}

// NewParser creates a new GBNF parser.
func NewParser() *Parser {
	return &Parser{
		SymbolIDs: make(map[string]int),
	}
}

func (p *Parser) getSymbolID(name string) int {
	if id, ok := p.SymbolIDs[name]; ok {
		return id
	}
	id := len(p.SymbolIDs)
	p.SymbolIDs[name] = id
	return id
}

func (p *Parser) generateSymbolID(baseName string) int {
	id := len(p.SymbolIDs)
	name := baseName + "_" + strconv.Itoa(id)
	p.SymbolIDs[name] = id
	return id
}

func (p *Parser) addRule(ruleID int, rule Rule) {
	for len(p.Rules) <= ruleID {
		p.Rules = append(p.Rules, nil)
	}
	p.Rules[ruleID] = rule
}

// Parse parses a GBNF grammar string and returns a Grammar ready for use.
func Parse(src string) (*Grammar, error) {
	p := NewParser()
	if err := p.parse(src); err != nil {
		return nil, err
	}

	rootID, ok := p.SymbolIDs["root"]
	if !ok {
		return nil, fmt.Errorf("grammar does not contain a 'root' rule")
	}

	// Check for left recursion
	nRules := len(p.Rules)
	visited := make([]bool, nRules)
	inProgress := make([]bool, nRules)
	mayBeEmpty := make([]bool, nRules)
	for i := 0; i < nRules; i++ {
		if visited[i] {
			continue
		}
		if detectLeftRecursion(p.Rules, i, visited, inProgress, mayBeEmpty) {
			return nil, fmt.Errorf("left recursion detected at rule index %d", i)
		}
	}

	g := &Grammar{
		Rules:          p.Rules,
		PartialNRemain: 0,
	}

	// Build initial stacks from root rule
	rule := p.Rules[rootID]
	idx := 0
	for {
		var stack Stack
		if idx < len(rule) && !isEndOfSequence(rule[idx]) {
			stack = append(stack, StackPos{rootID, idx})
		}
		g.advanceStack(stack, &g.Stacks)

		// scan to end of alternative
		for idx < len(rule) && !isEndOfSequence(rule[idx]) {
			idx++
		}
		if idx < len(rule) && rule[idx].Type == TypeAlt {
			idx++
		} else {
			break
		}
	}

	return g, nil
}

func (p *Parser) parse(src string) error {
	pos := skipSpace(src, 0, true)
	for pos < len(src) {
		var err error
		pos, err = p.parseRule(src, pos)
		if err != nil {
			return err
		}
	}

	// validate all rule refs
	for _, rule := range p.Rules {
		for _, elem := range rule {
			if elem.Type == TypeRuleRef {
				if int(elem.Value) >= len(p.Rules) || p.Rules[elem.Value] == nil {
					// find name for error
					for name, id := range p.SymbolIDs {
						if id == int(elem.Value) {
							return fmt.Errorf("undefined rule: '%s'", name)
						}
					}
					return fmt.Errorf("undefined rule at index %d", elem.Value)
				}
			}
		}
	}

	return nil
}

func (p *Parser) parseRule(src string, pos int) (int, error) {
	nameStart := pos
	nameEnd := parseName(src, pos)
	if nameEnd == pos {
		return 0, fmt.Errorf("expected rule name at position %d", pos)
	}
	name := src[nameStart:nameEnd]
	pos = skipSpace(src, nameEnd, false)

	if pos+2 >= len(src) || src[pos:pos+3] != "::=" {
		return 0, fmt.Errorf("expected '::=' at position %d", pos)
	}
	pos = skipSpace(src, pos+3, true)

	ruleID := p.getSymbolID(name)
	pos = p.parseAlternates(src, pos, name, ruleID, false)

	// skip newline
	if pos < len(src) {
		if src[pos] == '\r' {
			pos++
			if pos < len(src) && src[pos] == '\n' {
				pos++
			}
		} else if src[pos] == '\n' {
			pos++
		}
	}
	pos = skipSpace(src, pos, true)
	return pos, nil
}

func (p *Parser) parseAlternates(src string, pos int, ruleName string, ruleID int, isNested bool) int {
	var rule Rule
	pos = p.parseSequence(src, pos, ruleName, &rule, isNested)
	for pos < len(src) && src[pos] == '|' {
		rule = append(rule, Element{TypeAlt, 0})
		pos = skipSpace(src, pos+1, true)
		pos = p.parseSequence(src, pos, ruleName, &rule, isNested)
	}
	rule = append(rule, Element{TypeEnd, 0})
	p.addRule(ruleID, rule)
	return pos
}

func (p *Parser) parseSequence(src string, pos int, ruleName string, rule *Rule, isNested bool) int {
	lastSymStart := len(*rule)

	for pos < len(src) {
		c := src[pos]

		if c == '"' {
			// literal string
			pos++
			lastSymStart = len(*rule)
			for pos < len(src) && src[pos] != '"' {
				cp, newPos := parseCharEscape(src, pos)
				pos = newPos
				*rule = append(*rule, Element{TypeChar, cp})
			}
			if pos < len(src) {
				pos++ // skip closing "
			}
			pos = skipSpace(src, pos, isNested)

		} else if c == '[' {
			// character class
			pos++
			startType := TypeChar
			if pos < len(src) && src[pos] == '^' {
				startType = TypeCharNot
				pos++
			}
			lastSymStart = len(*rule)
			for pos < len(src) && src[pos] != ']' {
				cp, newPos := parseCharEscape(src, pos)
				pos = newPos

				elemType := TypeCharAlt
				if len(*rule) == lastSymStart {
					elemType = startType
				}
				*rule = append(*rule, Element{ElementType(elemType), cp})

				if pos < len(src)-1 && src[pos] == '-' && src[pos+1] != ']' {
					cp2, newPos2 := parseCharEscape(src, pos+1)
					pos = newPos2
					*rule = append(*rule, Element{TypeCharRngUp, cp2})
				}
			}
			if pos < len(src) {
				pos++ // skip ]
			}
			pos = skipSpace(src, pos, isNested)

		} else if isWordChar(c) {
			// rule reference
			nameEnd := parseName(src, pos)
			name := src[pos:nameEnd]
			refID := p.getSymbolID(name)
			pos = skipSpace(src, nameEnd, isNested)
			lastSymStart = len(*rule)
			*rule = append(*rule, Element{TypeRuleRef, uint32(refID)})

		} else if c == '(' {
			// grouping
			pos = skipSpace(src, pos+1, true)
			subRuleID := p.generateSymbolID(ruleName)
			pos = p.parseAlternates(src, pos, ruleName, subRuleID, true)
			lastSymStart = len(*rule)
			*rule = append(*rule, Element{TypeRuleRef, uint32(subRuleID)})
			if pos < len(src) && src[pos] == ')' {
				pos++
			}
			pos = skipSpace(src, pos, isNested)

		} else if c == '.' {
			// any char
			lastSymStart = len(*rule)
			*rule = append(*rule, Element{TypeCharAny, 0})
			pos = skipSpace(src, pos+1, isNested)

		} else if c == '*' {
			pos = skipSpace(src, pos+1, isNested)
			p.handleRepetition(rule, lastSymStart, ruleName, 0, -1)

		} else if c == '+' {
			pos = skipSpace(src, pos+1, isNested)
			p.handleRepetition(rule, lastSymStart, ruleName, 1, -1)

		} else if c == '?' {
			pos = skipSpace(src, pos+1, isNested)
			p.handleRepetition(rule, lastSymStart, ruleName, 0, 1)

		} else if c == '{' {
			pos = skipSpace(src, pos+1, isNested)
			// parse {m} or {m,} or {m,n}
			numStart := pos
			for pos < len(src) && src[pos] >= '0' && src[pos] <= '9' {
				pos++
			}
			minTimes, _ := strconv.Atoi(src[numStart:pos])
			pos = skipSpace(src, pos, isNested)

			maxTimes := -1 // -1 = same as min (exact repetition)
			if pos < len(src) && src[pos] == '}' {
				maxTimes = minTimes
				pos = skipSpace(src, pos+1, isNested)
			} else if pos < len(src) && src[pos] == ',' {
				pos = skipSpace(src, pos+1, isNested)
				if pos < len(src) && src[pos] >= '0' && src[pos] <= '9' {
					numStart2 := pos
					for pos < len(src) && src[pos] >= '0' && src[pos] <= '9' {
						pos++
					}
					maxTimes, _ = strconv.Atoi(src[numStart2:pos])
					pos = skipSpace(src, pos, isNested)
				} else {
					maxTimes = -1 // no max (unbounded)
				}
				if pos < len(src) && src[pos] == '}' {
					pos = skipSpace(src, pos+1, isNested)
				}
			}
			p.handleRepetition(rule, lastSymStart, ruleName, minTimes, maxTimes)

		} else {
			break
		}
	}
	return pos
}

// handleRepetition rewrites the last symbol(s) in the rule for repetition.
// maxTimes == -1 means unbounded.
func (p *Parser) handleRepetition(rule *Rule, lastSymStart int, ruleName string, minTimes, maxTimes int) {
	if lastSymStart >= len(*rule) {
		return // nothing to repeat
	}

	prevRule := make(Rule, len(*rule)-lastSymStart)
	copy(prevRule, (*rule)[lastSymStart:])

	if minTimes == 0 {
		*rule = (*rule)[:lastSymStart]
	} else {
		// repeat the previous elements (minTimes - 1) more times
		for i := 1; i < minTimes; i++ {
			*rule = append(*rule, prevRule...)
		}
	}

	noMax := maxTimes < 0 // unbounded
	nOpt := 1
	if !noMax {
		nOpt = maxTimes - minTimes
	}

	var lastRecRuleID int
	for i := 0; i < nOpt; i++ {
		recRule := make(Rule, len(prevRule))
		copy(recRule, prevRule)

		recRuleID := p.generateSymbolID(ruleName)
		if i > 0 || noMax {
			refID := recRuleID
			if !noMax {
				refID = lastRecRuleID
			}
			recRule = append(recRule, Element{TypeRuleRef, uint32(refID)})
		}
		recRule = append(recRule, Element{TypeAlt, 0})
		recRule = append(recRule, Element{TypeEnd, 0})
		p.addRule(recRuleID, recRule)
		lastRecRuleID = recRuleID
	}

	if nOpt > 0 {
		*rule = append(*rule, Element{TypeRuleRef, uint32(lastRecRuleID)})
	}
}

// ========== Parser helpers ==========

func isWordChar(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-' || (c >= '0' && c <= '9') || c == '_'
}

func parseName(src string, pos int) int {
	i := pos
	for i < len(src) && isWordChar(src[i]) {
		i++
	}
	return i
}

func skipSpace(src string, pos int, newlineOK bool) int {
	for pos < len(src) {
		c := src[pos]
		if c == ' ' || c == '\t' {
			pos++
		} else if c == '#' {
			// comment to end of line
			for pos < len(src) && src[pos] != '\n' && src[pos] != '\r' {
				pos++
			}
		} else if newlineOK && (c == '\n' || c == '\r') {
			pos++
		} else {
			break
		}
	}
	return pos
}

func parseCharEscape(src string, pos int) (uint32, int) {
	if pos >= len(src) {
		return 0, pos
	}

	if src[pos] == '\\' && pos+1 < len(src) {
		pos++
		c := src[pos]
		pos++
		switch c {
		case 'n':
			return '\n', pos
		case 'r':
			return '\r', pos
		case 't':
			return '\t', pos
		case '\\':
			return '\\', pos
		case '"':
			return '"', pos
		case '[':
			return '[', pos
		case ']':
			return ']', pos
		case 'x':
			// \xNN
			if pos+2 <= len(src) {
				val, err := strconv.ParseUint(src[pos:pos+2], 16, 32)
				if err == nil {
					return uint32(val), pos + 2
				}
			}
			return uint32(c), pos
		case 'u':
			// \uNNNN
			if pos+4 <= len(src) {
				val, err := strconv.ParseUint(src[pos:pos+4], 16, 32)
				if err == nil {
					return uint32(val), pos + 4
				}
			}
			return uint32(c), pos
		default:
			return uint32(c), pos
		}
	}

	// regular character (may be multi-byte UTF-8)
	r, size := utf8.DecodeRuneInString(src[pos:])
	if r == utf8.RuneError && size <= 1 {
		return uint32(src[pos]), pos + 1
	}
	return uint32(r), pos + size
}

// ========== Left recursion detection ==========

func detectLeftRecursion(rules []Rule, ruleIndex int, visited, inProgress, mayBeEmpty []bool) bool {
	if inProgress[ruleIndex] {
		return true
	}
	inProgress[ruleIndex] = true

	rule := rules[ruleIndex]

	// check if rule may produce empty string
	atRuleStart := true
	for i := 0; i < len(rule); i++ {
		if isEndOfSequence(rule[i]) {
			if atRuleStart {
				mayBeEmpty[ruleIndex] = true
				break
			}
			atRuleStart = true
		} else {
			atRuleStart = false
		}
	}

	// recurse into leftmost nonterminals
	recurse := true
	for i := 0; i < len(rule); i++ {
		if rule[i].Type == TypeRuleRef && recurse {
			if detectLeftRecursion(rules, int(rule[i].Value), visited, inProgress, mayBeEmpty) {
				return true
			}
			if !mayBeEmpty[int(rule[i].Value)] {
				recurse = false
			}
		} else if isEndOfSequence(rule[i]) {
			recurse = true
		} else {
			recurse = false
		}
	}

	inProgress[ruleIndex] = false
	visited[ruleIndex] = true
	return false
}

// ========== JSON Schema to GBNF ==========

// JSONSchemaToGBNF converts a simple JSON schema description to a GBNF grammar string.
// This supports basic types: object, array, string, number, integer, boolean, null.
// For objects, it supports "properties" and "required" fields.
func JSONSchemaToGBNF(schema map[string]interface{}) string {
	var b strings.Builder
	b.WriteString("root ::= ")
	writeSchemaRule(&b, schema, "root", true)
	b.WriteString("\n")

	// common rules
	b.WriteString(`
value ::= object | array | string | number | "true" | "false" | "null"

object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws

array ::= "[" ws (value ("," ws value)*)? "]" ws

string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

integer ::= ("-"? ([0-9] | [1-9] [0-9]*)) ws

boolean ::= ("true" | "false") ws

null ::= "null" ws

ws ::= | " " | "\n" [ \t]*
`)
	return b.String()
}

func writeSchemaRule(b *strings.Builder, schema map[string]interface{}, name string, isRoot bool) {
	typ, _ := schema["type"].(string)

	switch typ {
	case "object":
		props, hasProps := schema["properties"].(map[string]interface{})
		required := getStringSlice(schema, "required")
		requiredSet := make(map[string]bool)
		for _, r := range required {
			requiredSet[r] = true
		}

		if hasProps && len(props) > 0 {
			b.WriteString(`"{" ws `)
			first := true
			for propName := range props {
				if !first {
					b.WriteString(` "," ws `)
				}
				b.WriteString(`"\"` + propName + `\"" ":" ws `)
				// for simplicity, use 'value' for all property types
				b.WriteString("value")
				first = false
			}
			b.WriteString(` "}" ws`)
		} else {
			b.WriteString("object")
		}

	case "array":
		b.WriteString("array")
	case "string":
		b.WriteString("string")
	case "number":
		b.WriteString("number")
	case "integer":
		b.WriteString("integer")
	case "boolean":
		b.WriteString("boolean")
	case "null":
		b.WriteString("null")
	default:
		// generic value
		b.WriteString("value")
	}
}

func getStringSlice(m map[string]interface{}, key string) []string {
	arr, ok := m[key].([]interface{})
	if !ok {
		return nil
	}
	var result []string
	for _, v := range arr {
		if s, ok := v.(string); ok {
			result = append(result, s)
		}
	}
	return result
}

// Clone creates a deep copy of the grammar state.
func (g *Grammar) Clone() *Grammar {
	g2 := &Grammar{
		Rules:          g.Rules, // rules are static, can share
		PartialValue:   g.PartialValue,
		PartialNRemain: g.PartialNRemain,
	}
	g2.Stacks = make([]Stack, len(g.Stacks))
	for i, s := range g.Stacks {
		g2.Stacks[i] = make(Stack, len(s))
		copy(g2.Stacks[i], s)
	}
	return g2
}

// ========== Predefined Grammars ==========

// JSONGrammar is a GBNF grammar that constrains output to valid JSON.
const JSONGrammar = `root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

ws ::= | " " | "\n" [ \t]*
`