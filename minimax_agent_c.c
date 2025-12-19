#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>

// Game constants
#define NUM_ROWS 4
#define ROW_SIZE 5
#define MAX_CARDS 10
#define MAX_PLAYERS 10

// Game state structure
typedef struct {
    int table[NUM_ROWS][ROW_SIZE];
    int hands[MAX_PLAYERS][MAX_CARDS];
    int num_players;
} GameState;

// Cache entry structure
typedef struct {
    double score;
    int card_idx;
    int row_idx;
    int has_action;
} CacheEntry;

// Function declarations
static int compute_row_penalty(int* row);
static int resolve_card(int card, int row_to_take, int table[NUM_ROWS][ROW_SIZE]);
static void take_row(int card, int row_to_take, int table[NUM_ROWS][ROW_SIZE]);
static void get_best_row_order(int table[NUM_ROWS][ROW_SIZE], int* order);
static void copy_gamestate(GameState* dest, GameState* src);
static int gamestate_is_terminal(GameState* state);
static void gamestate_available_cards(GameState* state, int player_index, int* cards, int* count);
static void cache_init(void);
static unsigned long hash_string(const char* str, size_t len);
static CacheEntry* cache_get(const char* key, size_t key_len);
static void cache_put(const char* key, size_t key_len, CacheEntry entry);
static double max_node(GameState* state, int depth, double alpha, double beta, int player_index, int* best_card, int* best_row);
static double min_node(GameState* state, int depth, int my_card_idx, int my_row, double alpha, double beta, int player_index);
static void play_round(GameState* state, int my_card_idx, int my_row, int opp_card_idx, int opp_row, int player_index, GameState* next_state, double* penalties);
static double score_penalties(double* penalties, int player_index);

// Python interface
static PyObject* choose_action(PyObject* self, PyObject* args);

static PyMethodDef MinimaxAgentMethods[] = {
    {"choose_action", choose_action, METH_VARARGS, "Choose best action using minimax"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef minimax_agent_c_module = {
    PyModuleDef_HEAD_INIT,
    "minimax_agent_c",
    NULL,
    -1,
    MinimaxAgentMethods
};

PyMODINIT_FUNC PyInit_minimax_agent_c(void) {
    import_array();
    return PyModule_Create(&minimax_agent_c_module);
}

// Helper function implementations
static int compute_row_penalty(int* row) {
    int penalty = 0;
    for (int i = 0; i < ROW_SIZE; i++) {
        int card = row[i];
        if (card == 0) continue;
        if (card == 55) {
            penalty += 7;
        } else if (card % 11 == 0) {
            penalty += 5;
        } else if (card % 10 == 0) {
            penalty += 3;
        } else if (card % 5 == 0) {
            penalty += 2;
        } else {
            penalty += 1;
        }
    }
    return penalty;
}

static int resolve_card(int card, int row_to_take, int table[NUM_ROWS][ROW_SIZE]) {
    int end_cards[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++) {
        end_cards[i] = 0;
        for (int j = ROW_SIZE - 1; j >= 0; j--) {
            if (table[i][j] != 0) {
                end_cards[i] = table[i][j];
                break;
            }
        }
    }

    int diffs[NUM_ROWS];
    int can_place = 0;
    for (int i = 0; i < NUM_ROWS; i++) {
        diffs[i] = card - end_cards[i];
        if (diffs[i] > 0) can_place = 1;
    }

    if (!can_place) {
        // Must take a row - use the specified row_to_take
        take_row(card, row_to_take, table);
        return compute_row_penalty(table[row_to_take]);
    }

    // Find row with smallest positive diff (card > end_card)
    int best_row = -1;
    int min_diff = INT_MAX;
    for (int i = 0; i < NUM_ROWS; i++) {
        if (diffs[i] > 0 && diffs[i] < min_diff) {
            min_diff = diffs[i];
            best_row = i;
        }
    }

    // Count cards in the row
    int num_cards = 0;
    for (int i = 0; i < ROW_SIZE; i++) {
        if (table[best_row][i] != 0) num_cards++;
    }

    if (num_cards == ROW_SIZE) {
        // Row is full, take it
        take_row(card, best_row, table);
        return compute_row_penalty(table[best_row]);
    }

    // Place card
    table[best_row][num_cards] = card;
    return 0;
}

static void take_row(int card, int row_to_take, int table[NUM_ROWS][ROW_SIZE]) {
    table[row_to_take][0] = card;
    for (int i = 1; i < ROW_SIZE; i++) {
        table[row_to_take][i] = 0;
    }
}

static void get_best_row_order(int table[NUM_ROWS][ROW_SIZE], int* order) {
    int penalties[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++) {
        penalties[i] = compute_row_penalty(table[i]);
        order[i] = i;
    }

    // Simple bubble sort by penalty
    for (int i = 0; i < NUM_ROWS - 1; i++) {
        for (int j = 0; j < NUM_ROWS - i - 1; j++) {
            if (penalties[j] > penalties[j + 1] || (penalties[j] == penalties[j + 1] && order[j] > order[j + 1])) {
                int temp_penalty = penalties[j];
                penalties[j] = penalties[j + 1];
                penalties[j + 1] = temp_penalty;

                int temp_order = order[j];
                order[j] = order[j + 1];
                order[j + 1] = temp_order;
            }
        }
    }
}

static void copy_gamestate(GameState* dest, GameState* src) {
    memcpy(dest->table, src->table, sizeof(int) * NUM_ROWS * ROW_SIZE);
    memcpy(dest->hands, src->hands, sizeof(int) * MAX_PLAYERS * MAX_CARDS);
    dest->num_players = src->num_players;
}

static int gamestate_is_terminal(GameState* state) {
    for (int p = 0; p < state->num_players; p++) {
        for (int c = 0; c < MAX_CARDS; c++) {
            if (state->hands[p][c] != 0) return 0;
        }
    }
    return 1;
}

static void gamestate_available_cards(GameState* state, int player_index, int* cards, int* count) {
    *count = 0;
    for (int i = 0; i < MAX_CARDS; i++) {
        if (state->hands[player_index][i] != 0) {
            cards[*count] = i;
            (*count)++;
        }
    }
}

// Simple hash table for caching - simplified since depth is small
#define CACHE_SIZE 4096

typedef struct {
    char key[256];
    CacheEntry entry;
    int valid;
} CacheSlot;

static CacheSlot cache_slots[CACHE_SIZE];

static void cache_init(void) {
    memset(cache_slots, 0, sizeof(cache_slots));
}

static void cache_clear(void) {
    memset(cache_slots, 0, sizeof(cache_slots));
}

static unsigned long hash_string(const char* str, size_t len) {
    unsigned long hash = 5381;
    for (size_t i = 0; i < len && i < sizeof(cache_slots[0].key) - 1; i++) {
        hash = ((hash << 5) + hash) + str[i];
    }
    return hash;
}

static CacheEntry* cache_get(const char* key, size_t key_len) {
    unsigned long hash = hash_string(key, key_len);
    size_t index = hash % CACHE_SIZE;

    for (size_t i = 0; i < CACHE_SIZE; i++) {
        size_t probe_index = (index + i) % CACHE_SIZE;
        if (cache_slots[probe_index].valid &&
            strncmp(cache_slots[probe_index].key, key, key_len) == 0) {
            return &cache_slots[probe_index].entry;
        }
    }
    return NULL;
}

static void cache_put(const char* key, size_t key_len, CacheEntry entry) {
    unsigned long hash = hash_string(key, key_len);
    size_t index = hash % CACHE_SIZE;

    for (size_t i = 0; i < CACHE_SIZE; i++) {
        size_t probe_index = (index + i) % CACHE_SIZE;
        if (!cache_slots[probe_index].valid ||
            strncmp(cache_slots[probe_index].key, key, key_len) == 0) {
            strncpy(cache_slots[probe_index].key, key, sizeof(cache_slots[0].key) - 1);
            cache_slots[probe_index].key[key_len] = '\0';
            cache_slots[probe_index].entry = entry;
            cache_slots[probe_index].valid = 1;
            return;
        }
    }
}

static unsigned long hash_gamestate(GameState* state) {
    unsigned long hash = 5381;
    int* ptr = (int*)state->table;
    for (int i = 0; i < NUM_ROWS * ROW_SIZE; i++) {
        hash = ((hash << 5) + hash) + ptr[i];
    }
    ptr = (int*)state->hands;
    for (int i = 0; i < MAX_PLAYERS * MAX_CARDS; i++) {
        hash = ((hash << 5) + hash) + ptr[i];
    }
    return hash;
}

static double max_node(GameState* state, int depth, double alpha, double beta, int player_index, int* best_card, int* best_row) {
    if (depth == 0 || gamestate_is_terminal(state)) {
        *best_card = -1;
        *best_row = -1;
        return 0.0;
    }

    int my_card_count;
    int my_cards[MAX_CARDS];
    gamestate_available_cards(state, player_index, my_cards, &my_card_count);

    if (my_card_count == 0) {
        *best_card = -1;
        *best_row = -1;
        return 0.0;
    }

    // Check cache
    char cache_key[256];
    unsigned long state_hash = hash_gamestate(state);
    snprintf(cache_key, sizeof(cache_key), "max_%d_%d_%d_%lu", depth, player_index, state->num_players, state_hash);
    CacheEntry* cached = cache_get(cache_key, strlen(cache_key));
    if (cached) {
        *best_card = cached->card_idx;
        *best_row = cached->row_idx;
        return cached->score;
    }

    double best_score = -INFINITY;
    int best_action_card = -1;
    int best_action_row = -1;

    int row_order[NUM_ROWS];
    get_best_row_order(state->table, row_order);

    // Evaluate actions
    typedef struct {
        double penalty;
        int card_idx;
        int row_idx;
    } ActionCandidate;

    ActionCandidate candidates[MAX_CARDS * NUM_ROWS];
    int candidate_count = 0;

    for (int c = 0; c < my_card_count; c++) {
        int card_idx = my_cards[c];
        for (int r = 0; r < NUM_ROWS; r++) {
            int row_choice = row_order[r];

            GameState temp_state;
            copy_gamestate(&temp_state, state);
            double penalties[MAX_PLAYERS] = {0};

            play_round(&temp_state, card_idx, row_choice, -1, -1, player_index, &temp_state, penalties);

            double my_penalty = penalties[player_index];
            candidates[candidate_count].penalty = my_penalty;
            candidates[candidate_count].card_idx = card_idx;
            candidates[candidate_count].row_idx = row_choice;
            candidate_count++;
        }
    }

    // Sort candidates by penalty (bubble sort)
    for (int i = 0; i < candidate_count - 1; i++) {
        for (int j = 0; j < candidate_count - i - 1; j++) {
            if (candidates[j].penalty > candidates[j + 1].penalty) {
                ActionCandidate temp = candidates[j];
                candidates[j] = candidates[j + 1];
                candidates[j + 1] = temp;
            }
        }
    }

    // Select top candidates (zero penalty first, then up to half)
    ActionCandidate selected_candidates[MAX_CARDS * NUM_ROWS];
    int selected_count = 0;

    // First pass: zero penalty moves
    for (int i = 0; i < candidate_count; i++) {
        if (candidates[i].penalty == 0.0) {
            selected_candidates[selected_count++] = candidates[i];
        }
    }

    if (selected_count == 0) {
        // No zero penalty moves, take top half
        int limit = candidate_count / 2;
        if (limit < 1) limit = 1;
        for (int i = 0; i < limit && i < candidate_count; i++) {
            selected_candidates[selected_count++] = candidates[i];
        }
    }

    for (int i = 0; i < selected_count; i++) {
        int card_idx = selected_candidates[i].card_idx;
        int row_choice = selected_candidates[i].row_idx;

        double score = min_node(state, depth, card_idx, row_choice, alpha, beta, player_index);

        if (score > best_score) {
            best_score = score;
            best_action_card = card_idx;
            best_action_row = row_choice;
        }

        alpha = alpha > best_score ? alpha : best_score;
        if (beta <= alpha) {
            break;
        }
    }

    *best_card = best_action_card;
    *best_row = best_action_row;

    // Cache result
    CacheEntry entry = {best_score, best_action_card, best_action_row, 1};
    cache_put(cache_key, strlen(cache_key), entry);

    return best_score;
}

static double min_node(GameState* state, int depth, int my_card_idx, int my_row, double alpha, double beta, int player_index) {
    // Check cache
    char cache_key[256];
    unsigned long state_hash = hash_gamestate(state);
    snprintf(cache_key, sizeof(cache_key), "min_%d_%d_%d_%d_%d_%lu", depth, my_card_idx, my_row, player_index, state->num_players, state_hash);
    CacheEntry* cached = cache_get(cache_key, strlen(cache_key));
    if (cached) {
        return cached->score;
    }

    int opponent = (player_index + 1) % state->num_players;
    int opp_card_count;
    int opp_cards[MAX_CARDS];
    gamestate_available_cards(state, opponent, opp_cards, &opp_card_count);

    GameState next_state;
    double penalties[MAX_PLAYERS];

    if (opp_card_count == 0) {
        // Only we move
        play_round(state, my_card_idx, my_row, -1, -1, player_index, &next_state, penalties);
        double step_score = score_penalties(penalties, player_index);

        if (depth > 1 && !gamestate_is_terminal(&next_state)) {
            int dummy_card, dummy_row;
            double child_score = max_node(&next_state, depth - 1, alpha, beta, player_index, &dummy_card, &dummy_row);
            return step_score + child_score;
        }
        return step_score;
    }

    double worst_for_me = INFINITY;
    int row_order[NUM_ROWS];
    get_best_row_order(state->table, row_order);

    for (int c = 0; c < opp_card_count; c++) {
        int opp_card_idx = opp_cards[c];
        for (int r = 0; r < NUM_ROWS; r++) {
            int opp_row = row_order[r];

            play_round(state, my_card_idx, my_row, opp_card_idx, opp_row, player_index, &next_state, penalties);
            double step_score = score_penalties(penalties, player_index);

            double total_score;
            if (depth > 1 && !gamestate_is_terminal(&next_state)) {
                int dummy_card, dummy_row;
                double child_score = max_node(&next_state, depth - 1, alpha, beta, player_index, &dummy_card, &dummy_row);
                total_score = step_score + child_score;
            } else {
                total_score = step_score;
            }

            if (total_score < worst_for_me) {
                worst_for_me = total_score;
            }

            beta = beta < worst_for_me ? beta : worst_for_me;
            if (beta <= alpha) {
                CacheEntry entry = {worst_for_me, -1, -1, 0};
                cache_put(cache_key, strlen(cache_key), entry);
                return worst_for_me;
            }
        }
    }

    CacheEntry entry = {worst_for_me, -1, -1, 0};
    cache_put(cache_key, strlen(cache_key), entry);
    return worst_for_me;
}

static void play_round(GameState* state, int my_card_idx, int my_row, int opp_card_idx, int opp_row, int player_index, GameState* next_state, double* penalties) {
    copy_gamestate(next_state, state);

    int cards[MAX_PLAYERS] = {0};
    int rows[MAX_PLAYERS] = {0};

    // Set our move
    if (my_card_idx >= 0) {
        cards[player_index] = next_state->hands[player_index][my_card_idx];
        rows[player_index] = my_row;
        next_state->hands[player_index][my_card_idx] = 0;
    }

    // Set opponent's move
    if (opp_card_idx >= 0) {
        int opponent = (player_index + 1) % next_state->num_players;
        cards[opponent] = next_state->hands[opponent][opp_card_idx];
        rows[opponent] = opp_row;
        next_state->hands[opponent][opp_card_idx] = 0;
    }

    memset(penalties, 0, sizeof(double) * next_state->num_players);

    // Find active players
    int active_players[MAX_PLAYERS];
    int active_count = 0;
    for (int p = 0; p < next_state->num_players; p++) {
        if (cards[p] != 0) {
            active_players[active_count++] = p;
        }
    }

    if (active_count == 0) return;

    // Sort by card value
    for (int i = 0; i < active_count - 1; i++) {
        for (int j = 0; j < active_count - i - 1; j++) {
            if (cards[active_players[j]] > cards[active_players[j + 1]]) {
                int temp = active_players[j];
                active_players[j] = active_players[j + 1];
                active_players[j + 1] = temp;
            }
        }
    }

    // Resolve cards
    for (int i = 0; i < active_count; i++) {
        int player = active_players[i];
        penalties[player] = resolve_card(cards[player], rows[player], next_state->table);
    }
}

static double score_penalties(double* penalties, int player_index) {
    int opponent = (player_index + 1) % 2; // For simplicity, assume 2-player game
    return penalties[opponent] - penalties[player_index];
}

// Python interface function
static PyObject* choose_action(PyObject* self, PyObject* args) {
    PyObject* obs_dict;
    int depth;
    int player_index;

    if (!PyArg_ParseTuple(args, "Oii", &obs_dict, &depth, &player_index)) {
        return NULL;
    }

    // Extract numpy arrays from dict
    PyArrayObject* table_array = (PyArrayObject*)PyDict_GetItemString(obs_dict, "table");
    PyArrayObject* hands_array = (PyArrayObject*)PyDict_GetItemString(obs_dict, "hands");
    PyArrayObject* num_players_array = (PyArrayObject*)PyDict_GetItemString(obs_dict, "num_players");

    if (!table_array || !hands_array || !num_players_array) {
        PyErr_SetString(PyExc_ValueError, "Missing required observation fields");
        return NULL;
    }

    // Convert to C arrays
    GameState state;
    memset(&state, 0, sizeof(GameState));

    int* table_data = (int*)PyArray_DATA(table_array);
    for (int i = 0; i < NUM_ROWS; i++) {
        for (int j = 0; j < ROW_SIZE; j++) {
            state.table[i][j] = table_data[i * ROW_SIZE + j];
        }
    }

    int* hands_data = (int*)PyArray_DATA(hands_array);
    int num_players = *(int*)PyArray_DATA(num_players_array);
    state.num_players = num_players;

    for (int p = 0; p < state.num_players; p++) {
        for (int c = 0; c < MAX_CARDS; c++) {
            state.hands[p][c] = hands_data[p * MAX_CARDS + c];
        }
    }

    // Clear and initialize cache
    cache_clear();

    // Run minimax
    int best_card, best_row;
    max_node(&state, depth, -INFINITY, INFINITY, player_index, &best_card, &best_row);

    if (best_card == -1 || best_row == -1) {
        PyErr_SetString(PyExc_RuntimeError, "No valid action found");
        return NULL;
    }

    return Py_BuildValue("(ii)", best_card, best_row);
}