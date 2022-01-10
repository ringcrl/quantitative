#-*- coding:utf-8 -*-
from datetime import datetime, timedelta
from futu import *
import time
import numpy as np
import math
from multiprocessing import Pool
import re
from dotenv import dotenv_values
from sendmail import send_mail

config = dotenv_values(".env") 

# 日期计算
current_dt = time.strftime("%Y-%m-%d", time.localtime())
current_date = datetime.strptime(current_dt, '%Y-%m-%d')
previous_date = current_date - timedelta(days = 1)
curr_hour = time.localtime()[3]
curr_min = time.localtime()[4]
# 美国夏令时是中国21：30－4：00，非夏令时是22：30－5：00
open_time = {
    "h": 22,
    "m": 30,
}
close_time = {
    'h': 5,
    'm': 0,
}

GENERAL_MATCH = r'QQQ|DIA|SPY|UVXY'

freq = 'K_DAY' # K_DAY | K_60M

is_recall = config['is_recall'] == 'True'
is_send_email = config['is_send_email'] == 'True'
is_custom = config['is_custom'] == 'True'

holding_stocks = config.get('holding_stocks')
general_stocks = config.get('general_stocks')
watching_stocks = config.get('watching_stocks')
test_stocks = config.get('test_stocks')

holding_stocks = holding_stocks.split('|') if holding_stocks else []
general_stocks = general_stocks.split('|') if general_stocks else []
watching_stocks = watching_stocks.split('|') if watching_stocks else []
test_stocks = test_stocks.split('|') if test_stocks else []

# 自选股
custom_stocks = holding_stocks + general_stocks + watching_stocks + test_stocks
win_num = 0 # 统计获胜比例

# 动量轮动参数
N = 18 # 计算最新斜率 slope，拟合度 r2 参考最近 N 天，18
M = 200 # 计算最新标准分 zscore，rsrs_score 参考最近 M 天，600
RSRS_THRESHOLD = 0.4 # rsrs 标准分指标阈值
MEAN_DAY = 50 # 计算结束值，参考最近 MEAN_DAY
MEAN_DIFF_DAY = 5 # 计算初始值，参考(MEAN_DAY + MEAN_DIFF_DAY)天前，窗口为 MEAN_DIFF_DAY 的一段时间
RECALL_DAYS = 250 # 回测天数
STOP_LOSS = 0.98 # 止损点位
SUPPORT = '买点'
RESISTANCE = '卖点'
TWINE = '缠绕'

# 择时模块-计算综合信号，rsrs 信号算法参考优化说明，与其他值共同判断减少误差
def get_timing_signal(stock_data, slope_series):
    # 返回格式：stock_signal, rsrs_score, val_status

    rsrs_score = get_rsrs_score(stock_data, slope_series) # number
    volume_signal = get_volume_signal(stock_data) # VAL_UP | VAL_DOWN | VAL_STILL
    shooting_signal = get_shooting_signal(stock_data) # '' | TOP|num | BOTTOM|num
    stock_score = get_stock_score(stock_data) # float
    gmma_signal = get_gmma_signal(stock_data.close.values) # GMMA_UP | GMMA_DOWN | GMMA_TWINE
    
    return {
        "rsrs_score": round(rsrs_score, 2),
        "volume_signal": volume_signal,
        "shooting_signal": shooting_signal,
        "stock_score": stock_score,
        "gmma_signal": gmma_signal,
    }

# 计算 rsrs 信号：https://zhuanlan.zhihu.com/p/33501881
def get_rsrs_score(stock_data, slope_series):
    high_low_data = stock_data[-N:]
    intercept, slope, r2 = get_ols(high_low_data.low, high_low_data.high)
    slope_series.append(slope)
    rsrs_score = get_zscore(slope_series[-M:]) * r2
    return rsrs_score

# 成交量信号
def get_volume_signal(stock_data):
    mean_day_value = stock_data.turnover[-MEAN_DAY:].mean()
    mean_diff_day_value = stock_data.turnover[-MEAN_DIFF_DAY:].mean()
    return mean_diff_day_value / mean_day_value

# 射击之星和锤头线信号
def get_shooting_signal(stock_data):
    # 返回格式：None | 'TOP|num' | 'BOTTOM|num'
    len_rate = 2 # 一般2-3倍
    close_price = stock_data.close.values[-1]
    high_price = stock_data.high.values[-1]
    low_price = stock_data.low.values[-1]
    open_price = stock_data.open.values[-1]
    entity_len = abs(close_price - open_price)
    upline_len = 0
    downline_len = 0
    # 收盘价高于开盘价
    if close_price >= open_price:
        upline_len = abs(high_price - close_price)
        downline_len = abs(low_price - open_price)
    else:
        upline_len = abs(high_price - open_price)
        downline_len = abs(low_price - close_price)
    if upline_len >= entity_len * len_rate:
        return f'''TOP|{round(high_price, 2)}'''
    if downline_len >= entity_len * len_rate:
        return f'''BOTTOM|{round(low_price, 2)}'''
    
    else:
        return 'None'

def get_gmma_signal(close: np.array):
    ema10 = get_ema(close, 10)[-1]
    ema20 = get_ema(close, 20)[-1]
    ema30 = get_ema(close, 30)[-1]
    ema40 = get_ema(close, 40)[-1]
    ema50 = get_ema(close, 50)[-1]
    ema60 = get_ema(close, 60)[-1]

    if ema10 > ema20 > ema30 > ema40 > ema50 > ema60:
        return f'''GMMA_UP|{round(ema60 * STOP_LOSS, 2)}|{round(ema50, 2)}'''
    elif ema10 < ema20 < ema30 < ema40 < ema50 < ema60:
        return f'''GMMA_DOWN|{round(ema10 * STOP_LOSS, 2)}|{round(ema30, 2)}'''
    else:
        return f'''GMMA_TWINE|{round(ema10 * STOP_LOSS, 2)}|{round(ema10, 2)}'''

# 获取 EMA
def get_ema(close: np.array, timeperiod=5):
    res = []
    for i in range(len(close)):
        if i < 1:
            res.append(close[i])
        else:
            ema = (2 * close[i] + res[i-1] * (timeperiod-1)) / (timeperiod+1)
            res.append(ema)
    return np.array(res, dtype=np.double)

# 按长度分组
def cut(obj, sec):
    return [obj[i:i+sec] for i in range(0,len(obj),sec)]

# 根据市值，获取股票池
def get_market_cap(market_val=100000000000):
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
def get_stock_signals(stock_data):
    slope_series = initial_slope_series(stock_data, N, M)[:-1]
    return get_timing_signal(stock_data, slope_series)

# 运行回测
def recall(stock_code):
    monkey_base = 100000
    monkey_count = monkey_base
    stock_count = 0
    stock_data_all = get_code_data(stock_code)
    if stock_data_all is None or len(stock_data_all) < 900:
        return f'''error {stock_code}'''
    max_retreat = 0 # 最大回撤百分比
    before_total = 0 # 当前总额
    keep_stocks = 0 # 用于比对的股数，一开始躺平不动
    trade_num = 0 # 交易次数

    for i in range(RECALL_DAYS):
        before_day = RECALL_DAYS - i - 1
        split_a = -before_day-N-M
        split_b = -before_day
        stock_data = stock_data_all[split_a:split_b]
        if before_day == 0:
            stock_data = stock_data_all[split_a:]
        signal_str = op_signal(stock_data)
        close_price = stock_data.close.values[-1]
        is_buy = signal_str.startswith('BUY')
        is_sell = signal_str.startswith('SELL')
        curr_date = stock_data.time_key.values[-1][2:10]
        info = f'''{curr_date} {signal_str}'''

        if i == 0:
            keep_stocks = monkey_count / close_price

        if is_buy and monkey_count != 0:
            stock_count = monkey_count / close_price
            monkey_count = 0
            before_total = stock_count * close_price
            trade_num += 1
            print(info + ' 买入')
        elif is_sell and stock_count != 0:
            monkey_count = stock_count * close_price
            stock_count = 0
            # 当笔交易亏钱
            if monkey_count < before_total:
                curr_retreat = (before_total - monkey_count) / before_total * 100
                if curr_retreat > max_retreat:
                    max_retreat = curr_retreat
            trade_num += 1
            print(info + ' 卖出')
        else:
            print(info)
    final_close = stock_data_all.close.values[-1]
    total = monkey_count + stock_count * final_close
    keep_total = keep_stocks * final_close
    op_res = (total - monkey_base) / monkey_base * 100
    no_op_res = (keep_total - monkey_base) / monkey_base * 100
    is_win = op_res > no_op_res and op_res > 0
    op_res_str = round(op_res, 2)
    no_op_res_str = round(no_op_res, 2)
    max_retreat = round(max_retreat, 2)
    res = f'''{stock_code} 盈亏比：{op_res_str}% 躺平盈亏比：{no_op_res_str}% 最新单价：{final_close} 最大回撤：-{max_retreat}% 交易次数：{trade_num} 获胜：{is_win}'''
    print(res)
    return res

# 批量运行回测
def batch_recall(is_custom=True):
    stock_codes = []
    if is_custom:
        stock_codes = custom_stocks
    else:
        stock_codes = get_market_cap()
    
    info_list = []
    group = cut([i for i in stock_codes], 10)
    for group_stock_codes in group:
        p = Pool(len(group_stock_codes))
        res_list = p.map(recall, group_stock_codes)
        info_list += res_list
        p.close()
        p.join()
    filtered_info_list = []
    for stock_info in info_list:
        if stock_info.startswith('error'):
            continue
        filtered_info_list.append(stock_info)
    info = '\r\n'.join(filtered_info_list)
    win_len = len(re.findall('True', info))
    all_len = len(filtered_info_list)
    win_rate = round(win_len / all_len * 100, 2)
    msg_list = []
    msg_list.append(f'''{current_dt} 近{RECALL_DAYS}天回测结果 胜率：{win_rate}%({win_len}/{all_len})''')
    msg_list += filtered_info_list
    return msg_list

# 批量获取自选股信号
def batch_op_signal(is_custom=True):
    stock_codes = []
    if is_custom:
        stock_codes = custom_stocks
    else:
        stock_codes  = get_market_cap()
    
    info_list = []
    group = cut([i for i in stock_codes], 10)
    for group_stock_codes in group:
        p = Pool(len(group_stock_codes))
        res_list = p.map(get_stock_signal, group_stock_codes)
        info_list += res_list
        p.close()
        p.join()
    filtered_info_list = []
    for stock_info in info_list:
        if stock_info.startswith('error'):
            continue
        filtered_info_list.append(stock_info)

    msg_list = []
    general_list = [item for item in filtered_info_list if re.search(GENERAL_MATCH, item) is not None]
    custom_list = [item for item in filtered_info_list if re.search(GENERAL_MATCH, item) is None]
    list_up = [item for item in custom_list if SUPPORT in item]
    list_down = [item for item in custom_list if RESISTANCE in item]
    list_twine = [item for item in custom_list if TWINE in item]

    msg_list.append(f'''{current_dt} 操作信号''')
    msg_list.append('大盘趋势：')
    msg_list += general_list
    msg_list.append('上升趋势：')
    msg_list += list_up
    msg_list.append('下降趋势：')
    msg_list += list_down
    msg_list.append('缠绕趋势：')
    msg_list += list_twine

    return msg_list

def get_stock_signal(stock_code):
    stock_data = get_code_data(stock_code)
    return op_signal(stock_data)

def op_signal(stock_data):
    if len(stock_data.code.values) == 0:
        return 'error'

    stock_code = stock_data.code.values[0]
    stock_data = get_adjust_data(stock_data)

    a_s = get_stock_signals(stock_data[:-2])
    b_s = get_stock_signals(stock_data[:-1])
    c_s = get_stock_signals(stock_data)

    close_prices = f'''【{round(stock_data.close.values[-3], 2)}->{round(stock_data.close.values[-2], 2)}->{round(stock_data.close.values[-1], 2)}】'''

    latest_price = round(stock_data.close.values[-1], 2)

    op = get_op(latest_price, a_s, b_s, c_s)
    point_info = get_point_info(latest_price, a_s, b_s, c_s)

    vol = f'''vol({round(a_s['volume_signal'], 2)}->{round(b_s['volume_signal'], 2)}->{round(c_s['volume_signal'], 2)})'''
    gmma = f'''【{a_s['gmma_signal']}->{b_s['gmma_signal']}->{c_s['gmma_signal']}】'''
    rsrs = f'''rsrs({a_s['rsrs_score']}->{b_s['rsrs_score']}->{c_s['rsrs_score']})'''
    shoot = f'''shoot({a_s['shooting_signal']}->{b_s['shooting_signal']}->{c_s['shooting_signal']})'''

    res = f'''{op}{stock_code} {latest_price} {rsrs} {vol} {point_info}'''
    return res

def get_point_info(latest_price, a_s, b_s, c_s):
    gmma_info = c_s['gmma_signal'].split('|')
    
    point_signal = ''
    if gmma_info[0] == 'GMMA_UP':
        point_signal = SUPPORT
    elif gmma_info[0] == 'GMMA_DOWN':
        point_signal = RESISTANCE
    else:
        point_signal = TWINE
    key_money = ((float(gmma_info[2]) - latest_price) / latest_price) * 100
    stop_loss_monkey_per = ((float(gmma_info[1]) - latest_price) / latest_price) * 100
    
    if point_signal == TWINE:
        if key_money > 0:
            point_signal += '压力'
        else:
            point_signal += '支撑'
    point_info = f'''{point_signal}:{gmma_info[2]}({round(key_money, 2)}%)'''

    a_shooting = a_s['shooting_signal']
    b_shooting = b_s['shooting_signal']
    c_shooting = c_s['shooting_signal']
    if a_shooting != 'None':
        point_info += f''' {get_shooting_info(a_shooting)}'''
    if b_shooting != 'None':
        point_info += f''' {get_shooting_info(b_shooting)}'''
    if c_shooting !='None':
        point_info += f''' {get_shooting_info(c_shooting)}'''
    
    return point_info

def get_shooting_info(shooting_signal):
    [signal, price] = shooting_signal.split('|')
    if signal == 'TOP':
        return f'''量升不突破{price}卖出'''
    if signal == 'BOTTOM':
        return f'''跌破{price}卖出'''

def get_op(close_prices, a_s, b_s, c_s):
    BUY = 'BUY '
    SELL = 'SELL '
    NONE_INFO = ''

    # 止损
    # [gmma, stop_loss, key_point] = c_s['gmma_signal'].split('|')
    # if close_prices < stop_loss:
    #     return 'SELL'

    a_rsrs = a_s['rsrs_score']
    b_rsrs = b_s['rsrs_score']
    c_rsrs = c_s['rsrs_score']

    a_vol = a_s['volume_signal']
    b_vol = b_s['volume_signal']
    c_vol = c_s['volume_signal']

    gmma_signal = c_s['gmma_signal']

    # v1
    if c_rsrs >= RSRS_THRESHOLD:
        if c_vol >= 1:
            return BUY
        elif c_vol < 1:
            if not gmma_signal.startswith('GMMA_UP'):
                return SELL
            if a_vol < b_vol <c_vol:
                return BUY
    if c_rsrs <= -RSRS_THRESHOLD:
        if c_vol >= 1:
            return BUY
        elif c_vol < 1:
            if a_vol < b_vol < c_vol:
                return BUY
            else:
                return SELL

    # 无信号
    return NONE_INFO

# 开盘中成交量通过时间比例计算
def get_adjust_data(stock_data):
    # 直接返回
    return stock_data

    # 收盘中，直接用成交量
    if curr_hour > close_time['h'] and curr_hour < open_time['h']:
        return stock_data
    elif curr_hour == open_time['h'] and curr_min < open_time['m']:
        return stock_data
    
    # 开盘中，计算相对成交量
    trade_min_per_day = 6.5 * 60
    curr_trade_min = 0
    if curr_hour == open_time['h']:
        curr_trade_min = curr_min - open_time['m']
    elif curr_hour > open_time['h'] and curr_hour < 24:
        curr_trade_min = (curr_hour - open_time['h']) * 60 - open_time['m'] + curr_min
    elif curr_hour < open_time['h']:
        curr_trade_min = (24 - open_time['h']) * 60 - open_time['m'] + curr_hour * 60 + curr_min
    rate = trade_min_per_day / curr_trade_min
    stock_data['turnover'].values[-1] = stock_data['turnover'].values[-1] * rate
    return stock_data


if __name__ == "__main__":
    # print(f'\033[32m这是绿色字体\033[0m')
    # print(f'\033[31m这是红色字体\033[0m')

    msg_list = []
    if is_recall:
        msg_list = batch_recall(is_custom)        
    else:
        msg_list = batch_op_signal(is_custom)

    info = ''
    for msg in msg_list:
        if re.search(r'BUY|True', msg):
            info += f'\033[31m{msg}\033[0m\r\n'
        elif re.search(r'SELL', msg):
            info += f'\033[32m{msg}\033[0m\r\n'
        else:
            info += f'{msg}\r\n'
    print(info)

    if is_send_email:
        send_mail(msg_list)
