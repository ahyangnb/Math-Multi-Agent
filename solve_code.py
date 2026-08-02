
import random

def play_game(n):
    # B 有额外的一张鬼牌
    cards_A = set(range(1, n+1))
    cards_B = set(range(1, n+1)) | {'ghost'}
    
    while True:
        # A 抽 B 的牌
        drawn_card = random.choice(list(cards_B))
        if drawn_card == 'ghost':
            # A 抽到鬼牌，A 输
            return False
        if drawn_card in cards_A:
            # A 抽到了匹配的普通牌，丢弃两张牌
            cards_A.remove(drawn_card)
            cards_B.remove(drawn_card)
        else:
            # A 没有抽到匹配的牌，轮到 B 抽牌
            while True:
                if not cards_A:
                    # A 已经没有牌了，B 抽到鬼牌，B 输
                    return True
                drawn_card = random.choice(list(cards_A))
                if drawn_card in cards_B:
                    # B 抽到了匹配的牌，也丢弃两张牌
                    cards_A.remove(drawn_card)
                    cards_B.remove(drawn_card)
                    break
                # B 抽到了 A 的牌，但不是鬼牌，游戏继续，轮到 A
                return True

def simulate_games(n, trials=10000):
    wins = 0
    for _ in range(trials):
        if play_game(n):
            wins += 1
    return wins / trials

n_values = [31, 32, 999, 1000]
results = {n: simulate_games(n) for n in n_values}

for n, win_rate in results.items():
    print(f'n = {n}: A 的胜率约为 {win_rate:.4f}')
