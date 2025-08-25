import csv
import itertools
import time
from collections import defaultdict
import pandas as pd
from joblib import Parallel, delayed, dump, load
import math
import os

def load_transactions(path):
    """Carica dataset CSV in lista di frozenset"""
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
    # Raggruppa per transaction id e crea un frozenset per ciascun gruppo
    transactions = [
        frozenset(items)
        for items in df.groupby("tid")["item"].apply(list)
    ]
    return transactions

def save_memmap(transactions, filename="transactions.pkl"):
    """Salva le transazioni in formato joblib memmap"""
    dump(transactions, filename)
    return filename


def load_memmap(filename="transactions.pkl"):
    """Carica transazioni con memory mapping"""
    return load(filename, mmap_mode='r')


def support_worker_chunk(candidates_chunk, transactions):
    """Calcola il supporto per un chunk di candidati"""
    support = defaultdict(int)
    for transaction in transactions:
        for candidate in candidates_chunk:
            if candidate.issubset(transaction):
                support[candidate] += 1
    return support


def count_support_joblib_chunks(candidates, transactions, n_jobs, chunk_size):
    chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
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


def apriori_joblib_memmap(transactions, minsup, n_jobs=4, chunk_size=None):
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
        support_count = count_support_joblib_chunks(candidates, transactions, n_jobs, chunk_size_eff)
        Lk = filter_frequent(support_count, minsup, n_transactions)
        if not Lk:
            break
        L.append(Lk)
        support_data.update(Lk)
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


def generate_association_rules_joblib(frequent_itemsets, support_memmap, min_conf, n_jobs=4, chunk_size=None):
    all_itemsets = [itemset for k_itemsets in frequent_itemsets[1:] for itemset in k_itemsets.keys()]
    if not all_itemsets:
        return []

    if chunk_size is None:
        chunk_size_eff = len(all_itemsets) // n_jobs + 1
    else:
        chunk_size_eff = chunk_size
    chunks = [all_itemsets[i:i + chunk_size_eff] for i in range(0, len(all_itemsets), chunk_size_eff)]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(rules_single)(chunk, support_memmap, min_conf) for chunk in chunks
    )
    rules = [rule for partial in results for rule in partial]
    return rules


if __name__ == "__main__":
    dataset_csv = "retail_long.csv"
    memmap_transaction_file = "transactions.pkl"
    support_data_file = "support_data.pkl"

    # Se non esiste, crea memmap
    if not os.path.exists(memmap_transaction_file):
        print("📥 Caricamento CSV e salvataggio in memmap...")
        trans_list = load_transactions_from_long(dataset_csv)
        save_memmap(trans_list, memmap_transaction_file)

    # Carica da memmap
    transactions_memmap = load_memmap(memmap_transaction_file)
    minsup_values = [0.01, 0.02, 0.05]
    n_jobs_list = [2, 4, 8, 16]
    min_conf = 0.25
    chunk_size = [1, None]
    results_file = "results_joblib.csv"
    with open(results_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "minsup", "num_processes", "chunk_size", "apriori_time", "rules_time"])
        for ms in minsup_values:
            for n_jobs in n_jobs_list:
                for cs in chunk_size:
                    rules_times = []
                    start_apriori = time.perf_counter()
                    frequent_itemsets, support_data = apriori_joblib_memmap(transactions_memmap, ms, n_jobs, cs)
                    end_apriori = time.perf_counter()
                    apriori_time = end_apriori - start_apriori

                    save_memmap(support_data, support_data_file)
                    support_memmap = load_memmap(support_data_file)
                    for _ in range(10):
                        start_rules = time.perf_counter()
                        rules = generate_association_rules_joblib(frequent_itemsets, support_memmap, min_conf, n_jobs, cs)
                        end_rules = time.perf_counter()
                        rules_times.append(end_rules - start_rules)

                        avg_rules_time = sum(rules_times) / len(rules_times)

                        print(
                            f"Apriori found {sum(len(level) for level in frequent_itemsets)} frequent itemsets in {apriori_time:.6f} seconds (media su 10 iterazioni)")
                        print(f"Generated {len(rules)} rules in {avg_rules_time:.8f} seconds (media su 10 iterazioni)")

                        writer.writerow(
                            ["retail_long", ms, n_jobs, "max" if cs is None else f"{1}", f"{apriori_time:.6f}",
                             f"{avg_rules_time:.8f}"])
                        print("Results saved to", results_file)

    transactions = load_transactions('groceries - groceries.csv')
    with open(results_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "minsup", "num_processes", "chunk_size", "apriori_time", "rules_time"])
        for ms in minsup_values:
            for n_jobs in n_jobs_list:
                for cs in chunk_size:
                    rules_times = []
                    start_apriori = time.perf_counter()
                    frequent_itemsets, support_data = apriori_joblib_memmap(transactions_memmap, ms, n_jobs, cs)
                    end_apriori = time.perf_counter()
                    apriori_time = end_apriori - start_apriori

                    save_memmap(support_data, support_data_file)
                    support_memmap = load_memmap(support_data_file)
                    for _ in range(10):
                        start_rules = time.perf_counter()
                        rules = generate_association_rules_joblib(frequent_itemsets, support_memmap, min_conf, n_jobs,
                                                                  cs)
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


