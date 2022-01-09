#!/usr/bin/python
# -*- coding: UTF-8 -*-

import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
import time
from dotenv import dotenv_values
import re

config = dotenv_values(".env") 

sender = config['sender'] # 发件人邮箱账号
sender_pass = config['sender_pass'] # 发件人邮箱密码
receivers = config['receiver'].split('|') # 收件人邮箱账号

def send_mail(msg_list):
    ret = True
    try:
        current_dt = time.strftime("%Y-%m-%d", time.localtime())
        title = current_dt.split(" ")[0] + "操作"

        p_info = ''
        for msg in msg_list:
            if re.search(r'BUY|True', msg):
                p_info += f'''<p style="color: green;">{msg}</p>\r\n'''
            elif re.search(r'SELL', msg):
                p_info += f'''<p style="color: red;">{msg}</p>\r\n'''
            else:
                p_info += f'''<p style="color: black;">{msg}</p>\r\n'''

        html = f'''
<html>
<head></head>
<body>
    {p_info}
</body>
</html>
'''

        msg = MIMEText(html, 'html', 'utf-8')
        msg['From'] = formataddr(["chenng", sender])         # 发件人昵称
        msg['To'] = ','.join(receivers)             # 接收人昵称
        msg['Subject'] = title                              # 邮件的主题

        server = smtplib.SMTP_SSL("smtp.qq.com", 465)       # 发件人邮箱中的SMTP服务器，端口是465
        server.login(sender, sender_pass)  # 发件人邮箱账号、邮箱密码
        server.sendmail(sender, receivers, msg.as_string())  # 发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()  # 关闭连接
        print('邮件发送成功')
    except Exception as e:  # 如果 try 中的语句没有执行，则会执行下面的 ret = False
        ret = False
        print(e)
    return ret

if __name__ == "__main__":
    ret = send_mail("Test")
    if ret:
        print("邮件发送成功")
    else:
        print("邮件发送失败")
