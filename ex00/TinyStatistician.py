import array
import math

class TinyStatistician:

    def mean(self, x) -> None:
        if not x or not isinstance(x, (list, array.array)):
            return None
        
        total = 0.0
        count = 0
        for num in x:
            if not isinstance(num, (int, float)):
                return None
            total += num
            count += 1

        print(total / count)

    
    def median(self, x) -> None:
        if not x or not isinstance(x, (list, array.array)):
            return None
        
        sorted_x = sorted(x)
        length = len(sorted_x)

        if length % 2 == 0:
            big_i = length // 2
            small_i = big_i - 1
            middle = (sorted_x[big_i] + sorted_x[small_i]) / 2.0
        else:
            middle = sorted_x[length // 2]

        print(float(middle))

    def quartile(self, x) -> None:
        if not x or not isinstance(x, (list, array.array)):
            return None
        
        sorted_x = sorted(x)
        length = len(sorted_x)

        q1_exact = (length - 1) * 0.25
        if q1_exact.is_integer():
            q1 = sorted_x[int(q1_exact)]
        else:
            lower_index = int(q1_exact)
            upper_index = lower_index + 1
            distance = q1_exact - lower_index
            q1 = sorted_x[lower_index] + (sorted_x[upper_index] - sorted_x[lower_index]) * distance
        
        q3_exact = (length - 1) * 0.75
        if q3_exact.is_integer():
            q3 = sorted_x[int(q3_exact)]
        else:
            lower_index = int(q3_exact)
            upper_index = lower_index + 1
            distance = q3_exact - lower_index
            q3 = sorted_x[lower_index] + (sorted_x[upper_index] - sorted_x[lower_index]) * distance

        print([float(q1), float(q3)])


    def percentile(self, x, p) -> None:
        if not x or not isinstance(x, (list, array.array)):
            return None
        
        if not isinstance(p, (int, float)) or p < 0 or p > 100:
            return None
        
        sorted_x = sorted(x)
        length = len(sorted_x)
        
        exact_index = (length - 1) * p / 100
        lower_index = int(exact_index)
        
        if lower_index + 1 < length:
            upper_index = lower_index + 1
            distance = exact_index - lower_index
            percentile = sorted_x[lower_index] + (sorted_x[upper_index] - sorted_x[lower_index]) * distance
        else:
            percentile = sorted_x[lower_index]

        print(round(percentile, 1))

    def var(self, x) -> None:
        if not x or not isinstance(x, (list, array.array)):
            return None
        
        total = 0.0
        count = 0

        for num in x :
            if not isinstance(num, (int, float)):
                return None
            total += num
            count += 1

        mean = total / count
        var = 0.0
        for num in x :
            var += (num - mean) ** 2

        print(round(var / (count - 1), 1))
        
    def std(self, x) -> None:
        if not x or not isinstance(x, (list, array.array)):
            return None
        
        for num in x:
            if not isinstance(num, (int, float)):
                return None
        
        mean = sum(x) / len(x)
        var = sum((num - mean) ** 2 for num in x) / (len(x) - 1)
        std = math.sqrt(var)

        print(std)