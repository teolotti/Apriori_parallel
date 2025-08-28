import csv
import itertools
import time
from collections import defaultdict
import pandas as pd
from joblib import Parallel, delayed

# Carica le transazioni
def load_transactions(path):
    trans_df = pd.read_csv(path, header=None)
    trans_df.drop(trans_df.columns[0], axis=1, inplace=True)
    transactions = []
    for transaction in trans_df.values.tolist():
        items = [str(item) for item in transaction
                 if pd.notna(item) and str(item).lower() != 'nan' and 'item' not in str(item).lower()]
        if items:
            transactions.append(frozenset(items))
    return transactions

def load_transactions_from_long(path):
    df = pd.read_csv(path)
    transactions = [
        frozenset(items)
        for items in df.groupby("tid")["item"].apply(list)
    ]
    return transactions


def support_worker_chunk(candidates_chunk, transactions):
    support = defaultdict(int)
    for transaction in transactions:
        for candidate in candidates_chunk:
            if candidate.issubset(transaction):
                support[candidate] += 1
    return support

def count_support_joblib(candidates, transactions, n_jobs, chunk_size=None):
    chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
    results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(support_worker_chunk)(chunk, transactions) for chunk in chunks
    )
    merged = defaultdict(int)
    for partial in results:
        for itemset, count in partial.items():
            merged[itemset] += count
    return merged

def filter_frequent(support_count, minsup, n_transactions):
    return {
        itemset: count / n_transactions
        for itemset, count in support_count.items()
        if (count / n_transactions) >= minsup
    }

def apriori_joblib(transactions, minsup, n_jobs=4, chunk_size=None):
    n_transactions = len(transactions)
    items = sorted({item for transaction in transactions for item in transaction})
    L = []
    support_data = {}

    candidates = [frozenset([item]) for item in items]
    k = 1
    while candidates:
        if chunk_size is None:
            chunk_size_eff = len(candidates) // n_jobs + 1
        else:
            chunk_size_eff = chunk_size
        support_count = count_support_joblib(candidates, transactions, n_jobs, chunk_size_eff)
        Lk = filter_frequent(support_count, minsup, n_transactions)
        support_data.update(Lk)
        if not Lk:
            break
        L.append(Lk)
        prev_frequent = list(Lk.keys())
        candidates = list(set([i.union(j) for i in prev_frequent for j in prev_frequent if len(i.union(j)) == k + 1]))
        k += 1
    return L, support_data

def rules_single(support_data, min_conf, itemset_chunk=None):
    rules = []
    for itemset in itemset_chunk:
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if consequent:
                    conf = support_data[itemset] / support_data[antecedent]
                    if conf >= min_conf:
                        rules.append((antecedent, consequent, support_data[itemset], conf))
    return rules



def generate_association_rules_joblib(frequent_itemsets, support_data, min_conf, n_jobs=4, chunk_size=None):
    all_itemsets = [itemset for k_itemsets in frequent_itemsets[1:] for itemset in k_itemsets.keys()]
    if not all_itemsets:
        return []

    if chunk_size is None:
        chunk_size_eff = len(all_itemsets) // n_jobs + 1
    else:
        chunk_size_eff = chunk_size
    chunks = [all_itemsets[i:i + chunk_size_eff] for i in range(0, len(all_itemsets), chunk_size_eff)]

    results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(rules_single)(support_data, min_conf, chunk) for chunk in chunks
    )
    rules = [rule for partial in results for rule in partial]
    return rules

if __name__ == '__main__':
    # transactions = load_transactions('groceries - groceries.csv')
    transactions = load_transactions_from_long('datasets/retail_long.csv')  # Per testare con il dataset Kosarak
    minsup_values = [0.01, 0.02, 0.05]
    n_jobs_list = [2, 4, 8, 16, 32, 64]
    min_conf = 0.25
    chunk_size = [None]
    results_file = "results/results_joblib.csv"
    with open(results_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "minsup", "num_processes", "chunk_size", "apriori_time", "rules_time"])
        for ms in minsup_values:
            for n_jobs in n_jobs_list:
                for cs in chunk_size:
                    rules_times = []
                    start_apriori = time.perf_counter()
                    frequent_itemsets, support_data = apriori_joblib(transactions, ms, n_jobs, cs)
                    end_apriori = time.perf_counter()
                    apriori_time = end_apriori - start_apriori
                    for _ in range(10):
                        start_rules = time.perf_counter()
                        rules = generate_association_rules_joblib(frequent_itemsets, support_data, min_conf,n_jobs, cs)
                        end_rules = time.perf_counter()
                        rules_times.append(end_rules - start_rules)

                    avg_rules_time = sum(rules_times) / len(rules_times)

                    print(f"Apriori found {sum(len(level) for level in frequent_itemsets)} frequent itemsets in {apriori_time:.6f} seconds (media su 10 iterazioni)")
                    print(f"Generated {len(rules)} rules in {avg_rules_time:.8f} seconds (media su 10 iterazioni)")

                    writer.writerow(["retail_long", ms, n_jobs, "max" if cs is None else f"{1}", f"{apriori_time:.6f}", f"{avg_rules_time:.8f}"])
                    print("Results saved to", results_file)

    transactions = load_transactions('datasets/groceries - groceries.csv')
    with open(results_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "minsup", "num_processes", "chunk_size", "apriori_time", "rules_time"])
        for ms in minsup_values:
            for n_jobs in n_jobs_list:
                for cs in chunk_size:
                    rules_times = []
                    start_apriori = time.perf_counter()
                    frequent_itemsets, support_data = apriori_joblib(transactions, ms, n_jobs, cs)
                    end_apriori = time.perf_counter()
                    apriori_time = end_apriori - start_apriori
                    for _ in range(10):
                        start_rules = time.perf_counter()
                        rules = generate_association_rules_joblib(frequent_itemsets, support_data, min_conf, n_jobs, cs)
                        end_rules = time.perf_counter()
                        rules_times.append(end_rules - start_rules)

                    avg_rules_time = sum(rules_times) / len(rules_times)

                    print(
                        f"Apriori found {sum(len(level) for level in frequent_itemsets)} frequent itemsets in {apriori_time:.6f} seconds (media su 10 iterazioni)")
                    print(f"Generated {len(rules)} rules in {avg_rules_time:.8f} seconds (media su 10 iterazioni)")

                    writer.writerow(
                        ["groceries", ms, n_jobs, "max" if cs is None else f"{1}", f"{apriori_time:.6f}",
                         f"{avg_rules_time:.8f}"])
                    print("Results saved to", results_file)

