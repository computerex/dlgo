// Presets drawn from the real System One batteries (bench/systemone fixtures)
// plus a JSON-state example. Each exercises different question primitives.

import type { S1Question } from './types';

export interface S1Preset {
  name: string;
  blurb: string;
  stateText: string;
  stateIsJson: boolean;
  questions: S1Question[];
}

const q = (
  id: string,
  type: S1Question['type'],
  instructions: string,
  criteria?: Record<string, string> | string[],
): S1Question => {
  if (type === 'choice') {
    return {
      id, type, instructions,
      criteria: Object.entries(criteria as Record<string, string>).map(([key, label]) => ({ key, label })),
      levels: [],
    };
  }
  if (type === 'score') {
    return { id, type, instructions, criteria: [], levels: criteria as string[] };
  }
  return {
    id, type, instructions,
    criteria: Object.entries((criteria as Record<string, string>) ?? {}).map(([key, label]) => ({ key, label })),
    levels: [],
  };
};

export const S1_PRESETS: S1Preset[] = [
  {
    name: 'Pokemon battle',
    blurb: '2 choice + 1 noul — type-advantage reasoning',
    stateText:
      'You are in a Pokemon battle. The opponent\u0027s active Pokemon is Blastoise, a Water-type turtle that uses powerful Water attacks. Your active Pokemon is Charizard, a Fire/Flying-type at half health. Your party is: Charizard (Fire/Flying), Pikachu (Electric), Gengar (Ghost/Poison), and Snorlax (Normal).',
    stateIsJson: false,
    questions: [
      q('switch', 'choice', 'Which party member should you switch in to gain the best type advantage against the opponent\u0027s Water-type Blastoise?', {
        pikachu: 'Electric-type Pokemon; Electric attacks deal super-effective damage to Water Pokemon.',
        gengar: 'Ghost/Poison-type Pokemon; its attacks have no type advantage against Water.',
        snorlax: 'Normal-type Pokemon; its attacks have no type advantage against Water.',
      }),
      q('move_type', 'choice', 'Which move type deals the most damage to the opponent\u0027s Water-type Blastoise?', {
        electric: 'Electric moves deal super-effective (double) damage to Water Pokemon.',
        fire: 'Fire moves deal resisted (half) damage to Water Pokemon.',
        water: 'Water moves deal neutral damage to Water Pokemon.',
      }),
      q('charizard_water_weak', 'noul', 'Is your active Charizard, a Fire/Flying-type, weak to the opponent\u0027s Water attacks?', {
        true: 'Water attacks deal super-effective damage to Charizard.',
        false: 'Water attacks deal neutral, reduced, or no damage to Charizard.',
      }),
    ],
  },
  {
    name: 'Chess rules',
    blurb: 'Rules comprehension — choice/noul/score mix',
    stateText:
      'You are analyzing a game of standard international chess. The position is: White king on e1, White rook on a1, White pawn on e4; Black king on e8, Black queen on d8, Black knight on g8. It is White to move. Standard FIDE rules apply.',
    stateIsJson: false,
    questions: [
      q('knight_move', 'choice', 'How does a knight move on an empty board?', {
        l_shape: 'In an L shape: two squares in one direction then one square perpendicular.',
        diagonal: 'Any number of squares diagonally.',
        straight: 'Any number of squares in a straight line.',
        one_any: 'Exactly one square in any direction.',
      }),
      q('castling_condition', 'noul', 'In the given position, can White castle kingside this turn?'),
      q('promotion_rank', 'choice', 'To which rank must a White pawn reach to promote?', {
        rank8: 'The 8th rank.',
        rank1: 'The 1st rank.',
        rank5: 'The 5th rank.',
        any: 'Any rank.',
      }),
      q('en_passant', 'noul', 'Is capturing en passant a legal move in standard international chess?'),
      q('stalemate_result', 'choice', 'If a player has no legal move but their king is not in check, what is the result?', {
        draw: 'The game is a draw (stalemate).',
        loss: 'The player with no moves loses.',
        win: 'The player with no moves wins.',
        cont: 'The game continues.',
      }),
      q('board_score', 'score', 'How strong is a queen relative to the other chess pieces (in standard point values)?', [
        'Weak: worth about as much as a pawn.',
        'Below average: worth about as much as a knight.',
        'Strong: worth more than a bishop but slightly less than a rook.',
        'Very strong: worth substantially more than any other single piece.',
      ]),
    ],
  },
  {
    name: 'Yu-Gi-Oh duel',
    blurb: 'Game-rules battery — the ladder benchmark',
    stateText:
      'You are dueling in Yu-Gi-Oh. Your field has one face-up Attack Position monster (Summoned Skull, Level 6, 2500 ATK, DARK Fiend). Your hand contains: Blue-Eyes White Dragon (Level 8, LIGHT Dragon, 3000 ATK), Monster Reborn, Mystical Space Typhoon, Dark Hole, and Effect Veiler. Your opponent controls one face-up Attack Position monster (a 1800 ATK Warrior) and one Set Spell/Trap Card. Both players have 8000 Life Points. It is your Main Phase 1 and you have not yet Normal Summoned this turn.',
    stateIsJson: false,
    questions: [
      q('blue_eyes_normal_summon', 'noul', 'You control exactly one monster. You have Blue-Eyes White Dragon, a Level 8 monster, in your hand. Can you Normal Summon Blue-Eyes White Dragon this turn?'),
      q('destroys_all_monsters', 'choice', 'Which card destroys all monsters on the field?', {
        dark_hole: 'Dark Hole - destroys all monsters on the field.',
        raigeki: 'Raigeki - destroys all monsters your opponent controls.',
        mystic_typhoon: 'Mystical Space Typhoon - destroys one Spell or Trap Card.',
      }),
      q('blue_eyes_attribute', 'choice', 'What is Blue-Eyes White Dragon\u0027s Attribute?', {
        light: 'LIGHT',
        dark: 'DARK',
        earth: 'EARTH',
      }),
      q('negate_spell_half_lp', 'choice', 'Which card can negate the activation of a Spell or Trap Card by paying half your Life Points?', {
        solemn_judgment: 'Solemn Judgment',
        mystic_typhoon: 'Mystical Space Typhoon',
        dark_hole: 'Dark Hole',
      }),
      q('attack_strength_score', 'score', 'How strong is a 3000 ATK monster relative to typical Yu-Gi-Oh monsters?', [
        'Weak: below average ATK.',
        'Average: similar to most Level 4 monsters.',
        'Strong: above most monsters but not the highest.',
        'Very strong: among the highest ATK values in the game.',
      ]),
    ],
  },
  {
    name: 'GDPR comprehension',
    blurb: 'Long-doc grounded QA — includes a 5-level score (readout capacity)',
    stateText:
      'The General Data Protection Regulation (Regulation (EU) 2016/679), abbreviated GDPR, is a European Union regulation on information privacy in the European Union (EU) and the European Economic Area (EEA). It also governs the transfer of personal data outside the EU and EEA, and supersedes the Data Protection Directive 95/46/EC.\n\nThe European Parliament and Council of the European Union adopted the GDPR on 14 April 2016, to become effective on 25 May 2018. As an EU regulation (instead of a directive), the GDPR has direct legal effect and does not require transposition into national law.\n\nThe regulation applies if the data controller, or processor, or the data subject is based in the EU. It also applies to organisations based outside the EU if they collect or process personal data of individuals located inside the EU. Controllers must report personal data breaches to the supervisory authority without undue delay and, where feasible, not later than 72 hours after becoming aware of the breach, unless the breach is unlikely to result in a risk to the rights and freedoms of natural persons. Article 17 provides a right to erasure ("right to be forgotten"), and Article 20 provides a right to data portability. Administrative fines of up to EUR 20 million, or in the case of an undertaking, up to 4% of the total worldwide annual turnover of the preceding financial year, whichever is higher, apply to infringements of the basic principles for processing, data subjects\u0027 rights, and transfers. Records of processing activities, data protection impact assessments, and (for larger processors) a designated data protection officer are required; consent must be specific, freely given, plainly worded, and unambiguous \u2014 pre-ticked boxes are a violation.',
    stateIsJson: false,
    questions: [
      q('instrument_type', 'choice', 'What kind of EU legal instrument is the GDPR?', {
        regulation: 'Directly binding law in all member states, no national implementation needed.',
        directive: 'Sets goals that member states implement through national law.',
        treaty: 'An international treaty between states.',
        recommendation: 'Non-binding guidance.',
      }),
      q('max_fine', 'choice', 'What is the maximum administrative fine for the most serious infringements?', {
        twenty_m_4pct: 'Up to EUR 20 million or 4% of annual worldwide turnover, whichever is greater.',
        ten_m_2pct: 'Up to EUR 10 million or 2% of annual worldwide turnover, whichever is greater.',
        fixed_cap: 'A fixed amount not tied to turnover.',
        no_fines: 'The GDPR provides no administrative fines.',
      }),
      q('breach_72h', 'noul', 'Must a personal data breach be reported to the supervisory authority within 72 hours?'),
      q('right_erasure', 'noul', 'Does the regulation grant individuals a right to erasure of their personal data?'),
      q('pre_ticked_consent', 'noul', 'Can valid consent be obtained through pre-ticked boxes or inactivity?'),
      q('compliance_burden', 'score', 'How heavy is the compliance burden the GDPR places on organisations?', [
        'Negligible: no meaningful obligations.',
        'Light: a few notices and disclosures.',
        'Moderate: documented processes and some dedicated roles for larger processors.',
        'Heavy: records, impact assessments, officers, and breach procedures for many organisations.',
        'Extreme: obligations so demanding that ordinary organisations cannot fully comply.',
      ]),
    ],
  },
  {
    name: 'Ticket triage (JSON state)',
    blurb: 'JSON-object state + a 5-option choice (readout capacity boundary)',
    stateText: JSON.stringify(
      {
        ticket: 'TKT-4417',
        product: 'CloudSync Pro',
        customer_since: '2023-06-01',
        plan: 'Business',
        message:
          'Sync has been failing on all my devices since yesterday afternoon. I run a small agency and we bill from this data \u2014 three commits lost already. This is the second outage this quarter. If it is not fixed today I need to know about refunds.',
        contacts_last_30d: 6,
        escalations: 1,
      },
      null,
      2,
    ),
    stateIsJson: true,
    questions: [
      q('category', 'choice', 'Which single category best classifies this ticket?', {
        billing_refund: 'Customer requests money back or disputes a charge.',
        technical_outage: 'A product defect or outage the customer cannot work around.',
        account_access: 'Login, permissions, or account recovery problems.',
        feature_request: 'Customer asks for new or different functionality.',
        general_question: 'Anything that does not fit the other categories.',
      }),
      q('refund_eligible', 'noul', 'Based on the message, is the customer explicitly raising the possibility of a refund?'),
      q('urgency', 'score', 'How urgent is a response to this ticket?', [
        'Low: general inquiry, customer is patient.',
        'Moderate: inconvenience, but workarounds exist.',
        'High: blocked work, time pressure stated.',
        'Critical: revenue-impacting outage with escalation threats.',
      ]),
    ],
  },
  {
    name: 'Single question (legacy shape)',
    blurb: 'One choice — exercises the single-question legacy response fields',
    stateText: 'Customer message: "Honestly, after the last two support calls I expected better. The product itself is fine, but getting a human took three days both times."',
    stateIsJson: false,
    questions: [
      q('sentiment', 'choice', 'What is the overall sentiment of this customer message?', {
        positive: 'Satisfied with the product and service.',
        mixed: 'Positive on the product but negative on the service experience.',
        negative: 'Dissatisfied with both product and service.',
      }),
    ],
  },
];
