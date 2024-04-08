from TinyStatistician import TinyStatistician

if __name__ == "__main__":

    a = [1, 42, 300, 10, 59]
    
    TinyStatistician().mean(a) # 82.4

    TinyStatistician().median(a) # 42.0

    TinyStatistician().quartile(a) # [10.0, 59.0]

    TinyStatistician().percentile(a, 10)  # 4.6

    TinyStatistician().percentile(a, 15)  # 6.4

    TinyStatistician().percentile(a, 20)  # 8.2

    TinyStatistician().var(a)  #15349.3

    TinyStatistician().std(a)  #123.89229193133849
