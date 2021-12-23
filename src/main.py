'''
优化说明:
1.使用修正标准分
    rsrs_score的算法有：
        仅斜率slope，效果一般；
        仅标准分zscore，效果不错；
        修正标准分 = zscore * r2，效果最佳;
        右偏标准分 = 修正标准分 * slope，效果不错。
2.将原策略的每次持有两只etf改成只买最优的一个，收益显著提高
3.将每周调仓换成每日调仓，收益显著提高
4.因为交易etf，所以手续费设为万分之三，印花税设为零，未设置滑点
5.修改股票池中候选etf，删除银行，红利等收益较弱品种，增加纳指etf以增加不同国家市场间轮动的可能性
6.根据研报，默认参数介已设定为最优
7.加入防未来函数
8.增加择时与选股模块的打印日志，方便观察每笔操作依据
'''

#-*- coding:utf-8 -*-
from datetime import datetime, timedelta
from futu import *
import time
import numpy as np
import math
from multiprocessing import Pool

# 日期计算
current_dt = time.strftime("%Y-%m-%d", time.localtime())
current_date = datetime.strptime(current_dt, '%Y-%m-%d')
previous_date = current_date - timedelta(days = 1)
curr_hour = time.localtime()[3]

freq = 'K_DAY' # K_DAY | K_60M

# 自选股
custom_stocks = [
    'US.QQQ', # 纳指
    'US.DIA', # 道指
    'US.SPY', # 标普
    'US.TSLA',
    'US.NVDA',
    'US.AAPL',
    'US.GOOGL',
    'US.VZ',
    'US.BABA',
    'US.BILI',
    'US.SE',
    'US.PDD',
    'US.AMC',
    'US.GME',
    'HK.00700',
]

# 动量轮动参数
N = 18 # 计算最新斜率 slope，拟合度 r2 参考最近 N 天，18
M = 100 # 计算最新标准分 zscore，rsrs_score 参考最近 M 天，600
score_threshold = 0.7 # rsrs 标准分指标阈值
mean_day = 30 # 计算结束值，参考最近 mean_day
mean_diff_day = 3 # 计算初始值，参考(mean_day + mean_diff_day)天前，窗口为 mean_diff_day 的一段时间
volume_padding = 0.02 # 成交量差距百分比
recall_days = 150 # 回测天数

# 择时模块-计算综合信号，rsrs 信号算法参考优化说明，与其他值共同判断减少误差
def get_timing_signal(stock_data, mean_day, mean_diff_day, N, slope_series):
    # 当前值
    curr_val = stock_data.turnover[-mean_day:].mean()
    # 之前值
    before_val = stock_data.turnover[-mean_day-mean_diff_day:-mean_diff_day].mean()
    # 计算 rsrs 信号
    high_low_data = stock_data[-N:]
    intercept, slope, r2 = get_ols(high_low_data.low, high_low_data.high)
    slope_series.append(slope)
    rsrs_score = get_zscore(slope_series[-M:]) * r2

    diff_volume = curr_val / before_val

    if rsrs_score >= score_threshold:
        if diff_volume >= 1 + volume_padding:
            return 'BUY', rsrs_score, 'VAL_UP'
        elif diff_volume <= 1 - volume_padding:
            return 'KEEP', rsrs_score, 'VAL_DOWN'
        else:
            return 'SELL', rsrs_score, 'VAL_STILL'
    elif rsrs_score <= -score_threshold:
        if diff_volume >= 1 + volume_padding:
            return 'BUY', rsrs_score, 'VAL_UP'
        elif diff_volume <= 1 - volume_padding:
            return 'KEEP', rsrs_score, 'VAL_DOWN'
        else:
            return 'SELL', rsrs_score, 'VAL_STILL'
    else:
        if diff_volume >= 1 + volume_padding:
            return 'KEEP', rsrs_score, 'VAL_UP'
        elif diff_volume <= 1 - volume_padding:
            return 'KEEP', rsrs_score, 'VAL_DOWN'
        else:
            return 'SELL', rsrs_score, 'VAL_STILL'

# 根据市值，获取股票池
def get_market_cap(market_val=50000000000):
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    simple_filter = SimpleFilter()
    simple_filter.stock_field = StockField.MARKET_VAL
    simple_filter.filter_min = market_val
    simple_filter.is_no_filter = False

    stock_list = list()
    nBegin = 0
    last_page = False
    ret_list = list()
    while not last_page:
        nBegin += len(ret_list)
        ret, data = quote_ctx.get_stock_filter(market=Market.US, filter_list=simple_filter, begin=nBegin)  # 对香港市场的股票做简单和财务筛选
        if ret == RET_OK:
            last_page, all_count, ret_list = data
            print('all count = ', all_count)
            
            for item in ret_list:
                stock_list.append(item.stock_code)
        else:
            print('error: ', data)
        time.sleep(1)

    quote_ctx.close()
    return stock_list

# 根据股票代码获取历史K数据
def get_code_data(code, freq=freq, count=1000):
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    ret_sub, err_message = quote_ctx.subscribe([code], [freq], subscribe_push=False)
    if ret_sub == RET_OK:
        ret, data = quote_ctx.get_cur_kline(code, count, freq, AuType.QFQ)
        quote_ctx.close()
        if ret == RET_OK:
            return data
        else:
            print(ret)
            return None
    else:
        quote_ctx.close()
        print(err_message)
        return None

# 获取股票得分
def get_stock_score(stock_data):
        # 收盘价
    y = stock_data['log'] = np.log(stock_data.close)
    # 分析的数据个数（天）
    x = stock_data['num'] = np.arange(stock_data.log.size)
    # 拟合 1 次多项式
    # y = kx + b, slope 为斜率 k，intercept 为截距 b
    slope, intercept = np.polyfit(x, y, 1)
    # (e ^ slope) ^ 250 - 1
    annualized_returns = math.pow(math.exp(slope), 250) - 1
    r_squared = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
    score = annualized_returns * r_squared
    return score

# 择时模块-计算线性回归统计值
# 对输入的自变量每日最低价 x(series) 和因变量每日最高价 y(series) 建立 OLS 回归模型,返回元组(截距,斜率,拟合度)
# R2 统计学线性回归决定系数，也叫判定系数，拟合优度。
# R2 范围 0 ~ 1，拟合优度越大，自变量对因变量的解释程度越高，越接近 1 越好。
# 公式说明： https://blog.csdn.net/snowdroptulip/article/details/79022532
#           https://www.cnblogs.com/aviator999/p/10049646.html
def get_ols(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
    return (intercept, slope, r2)

# 择时模块-计算标准分，通过斜率列表计算并返回截至回测结束日的最新标准分
def get_zscore(slope_series):
    mean = np.mean(slope_series)
    std = np.std(slope_series)
    return (slope_series[-1] - mean) / std

# 择时模块-设定初始斜率序列，通过前 M 日最高最低价的线性回归计算初始的斜率，返回斜率的列表
def initial_slope_series(general_stock, N, M):
    return [get_ols(general_stock.low[i:i+N], general_stock.high[i:i+N])[1] for i in range(M)]

# 获取股票信号
def get_stock_signal(stock_data):
    slope_series = initial_slope_series(stock_data, N, M)[:-1]
    return get_timing_signal(stock_data, mean_day, mean_diff_day, N, slope_series)

# 运行回测
def recall(stock_code):
    monkey_count = 100000
    stock_count = 0
    stock_data_all = get_code_data(stock_code)

    for i in range(recall_days):
        before_day = recall_days - i
        stock_data = stock_data_all[-before_day-N-M:-before_day]
        timing_signal, rsrs_score, val_status = get_stock_signal(stock_data)
        close_price = stock_data.close[-1:].mean()
        is_buy = timing_signal == 'BUY'
        is_sell = timing_signal == 'SELL'
        info = f'''{stock_code} {i}/{recall_days} {timing_signal} {format(rsrs_score, '.2f')} {val_status} {format(close_price, '.2f')}'''
        if is_buy:
            if monkey_count != 0:
                stock_count = monkey_count / close_price
                monkey_count = 0
                print(info + ' 买入')
            else:
                print(info)
                pass
        elif is_sell:
            if stock_count != 0:
                monkey_count = stock_count * close_price
                stock_count = 0
                print(info + ' 卖出')
            else:
                print(info)
                pass
        else:
            print(info)
            pass
    final_close = stock_data_all.close[-1:].mean()
    total = monkey_count + stock_count * final_close
    res = f'''{stock_code} 总价：{format(total, '.2f')} 最新单价：{final_close}'''
    print(res)
    return res

# 批量运行回测
def batch_recall():
    p = Pool(len(custom_stocks))
    res = p.map(recall, custom_stocks)    
    p.close()
    p.join()
    info = '\r\n'.join(res)
    msg = f'''
{current_dt} 回测结果
{info}
'''
    print(msg)

# 批量获取自选股信号
def batch_op_signal():
    p = Pool(len(custom_stocks))
    res = p.map(op_signal, custom_stocks)
    p.close()
    p.join()
    info = '\r\n'.join(res)
    msg = f'''
{current_dt} 操作信号
{info}
'''
    print(msg)


def op_signal(stock_code):
    stock_data = get_code_data(stock_code)

    # 开盘中去掉今天的数据，防止成交量干扰
    if curr_hour >= 22 or curr_hour < 5:
        stock_data = stock_data[:-1]

    before_stock_signal, before_rsrs_score, before_val_status = get_stock_signal(stock_data[:-1])
    curr_stock_signal, curr_rsrs_score, curr_val_status = get_stock_signal(stock_data)
    prefix = ''
    if before_stock_signal != curr_stock_signal:
        if curr_stock_signal == 'BUY':
            prefix = '买 '
        elif curr_stock_signal == 'SELL':
            prefix = '卖 '
        else:
            prefix = '观察 '
    res = f'''{prefix}{stock_code} 【{before_stock_signal}->{curr_stock_signal}】 【{format(curr_rsrs_score, '.2f')} {curr_val_status}】'''
    return res

if __name__ == "__main__":
    # batch_recall()
    batch_op_signal()

    # recall('US.AMC')
