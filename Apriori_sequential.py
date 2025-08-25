import csv
import itertools
import time
from collections import defaultdict
from turtledemo.penrose import start
import pandas as pd


# Carica le transazioni da file (una transazione per riga, item separati da spazio)
def load_transactions(path):
    trans_df = pd.read_csv(path, header=None)
    trans_df.drop(trans_df.columns[0], axis=1, inplace=True)
    transactions = []
    for transaction in trans_df.values.tolist():
        items = [str(item) for item in transaction if pd.notna(item) and str(item).lower() != 'nan' and 'item' not in str(item).lower()]
        if items:
            transactions.append(frozenset(items))
    return transactions

def load_transactions_from_long(path):
    df = pd.read_csv(path)
    # Raggruppa per transaction id e crea un frozenset per ciascun gruppo
    transactions = [
        frozenset(items)
        for items in df.groupby("tid")["item"].apply(list)
    ]
    return transactions

# Conta il supporto degli itemset nel dataset
def count_support(candidates, transactions):
    support = defaultdict(int)
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                support[candidate] += 1
    return support

# Filtra itemset con supporto >= minsup
def filter_frequent(support_count, minsup, n_transactions):
    return {
        itemset: count / n_transactions
        for itemset, count in support_count.items()
        if (count / n_transactions) >= minsup
    }

def apriori(transactions, minsup):
    n_transactions = len(transactions)
    items = sorted({item for transaction in transactions for item in transaction})
    L = []
    support_data = {}

    # Itemset di lunghezza 1
    candidates = [frozenset([item]) for item in items]
    k = 1
    while candidates:
        support_count = count_support(candidates, transactions)
        Lk = filter_frequent(support_count, minsup, n_transactions)
        if not Lk:
            break
        L.append(Lk)
        support_data.update(Lk)
        # Genera nuovi candidati di lunghezza k+1
        prev_frequent = list(Lk.keys())
        candidates = [i.union(j) for i in prev_frequent for j in prev_frequent if len(i.union(j)) == k + 1]
        candidates = list(set(candidates))
        k += 1
    return L, support_data

def generate_association_rules(frequent_itemsets, support_data, min_conf):
    rules = []
    for k_itemsets in frequent_itemsets[1:]:  # salta i singoli item
        for itemset in k_itemsets.keys():
            for i in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    if consequent:
                        conf = support_data[itemset] / support_data[antecedent]
                        if conf >= min_conf:
                            rules.append((antecedent, consequent, support_data[itemset], conf))
    return rules

if __name__ == '__main__':

    results_file = "results_sequential.csv"
    # transactions = load_transactions('groceries - groceries.csv')
    transactions = load_transactions_from_long('retail_long.csv')  # Per testare con il dataset retail_long

    minsup_values = [0.01, 0.02, 0.05]
    minconf = 0.25

    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "minsup", "apriori_time", "rules_time"])

        for ms in minsup_values:
            rules_times = []

            start_apriori = time.perf_counter()
            frequent_itemsets, support_data = apriori(transactions, ms)
            end_apriori = time.perf_counter()
            apriori_time = end_apriori - start_apriori
            for _ in range(10):
                start_rules = time.perf_counter()
                rules = generate_association_rules(frequent_itemsets, support_data, minconf)
                end_rules = time.perf_counter()
                rules_times.append(end_rules - start_rules)

            avg_rules_time = sum(rules_times) / len(rules_times)

            print(f"Apriori found {sum(len(level) for level in frequent_itemsets)} frequent itemsets in {apriori_time:.6f} seconds (media su 10 iterazioni)")
            print(f"Generated {len(rules)} rules in {avg_rules_time:.8f} seconds (media su 10 iterazioni)")

            writer.writerow(["retail_long", ms, f"{apriori_time:.6f}", f"{avg_rules_time:.8f}"])
            print("Results saved to", results_file)

    transactions = load_transactions('groceries - groceries.csv')

    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)

        for ms in minsup_values:
            rules_times = []

            start_apriori = time.perf_counter()
            frequent_itemsets, support_data = apriori(transactions, ms)
            end_apriori = time.perf_counter()
            apriori_time = end_apriori - start_apriori
            for _ in range(10):
                start_rules = time.perf_counter()
                rules = generate_association_rules(frequent_itemsets, support_data, minconf)
                end_rules = time.perf_counter()
                rules_times.append(end_rules - start_rules)

            avg_rules_time = sum(rules_times) / len(rules_times)

            print(
                f"Apriori found {sum(len(level) for level in frequent_itemsets)} frequent itemsets in {apriori_time:.6f} seconds (media su 10 iterazioni)")
            print(f"Generated {len(rules)} rules in {avg_rules_time:.8f} seconds (media su 10 iterazioni)")

            writer.writerow(["groceries", ms, f"{apriori_time:.6f}", f"{avg_rules_time:.8f}"])
            print("Results saved to", results_file)

