#coding=utf-8
__author__ = 'jinhang'
from PIL import Image
from PIL import ImageFilter
import urllib
import urllib2
import re
import json
import tempfile
import os
import ssl
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.89 Safari/537.36"
#pic_url = "https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=login&rand=sjrand&0.21191171556711197"
pic_url = "C://76754b27584233c2287986dc0577854a_b.jpg
# google 的 pytesseract使用
def ocr_question_extract(im):
    # git@github.com:madmaze/pytesseract.git
    global pytesseract
    try:
        import pytesseract
    except:
        print "[ERROR] pytesseract not installed"
        return
    im = im.crop((127, 3, 260, 22))
    im = pre_ocr_processing(im)
    # im.show()
    return pytesseract.image_to_string(im, lang='chi_sim').strip()
# 二值化 加锐化
def pre_ocr_processing(im):
    im = im.convert("RGB")
    width, height = im.size
    white = im.filter(ImageFilter.BLUR).filter(ImageFilter.MaxFilter(23))
    grey = im.convert('L')
    impix = im.load()
    whitepix = white.load()
    greypix = grey.load()

    for y in range(height):
        for x in range(width):
            greypix[x,y] = min(255, max(255 + impix[x,y][0] - whitepix[x,y][0],
                                        255 + impix[x,y][1] - whitepix[x,y][1],
                                        255 + impix[x,y][2] - whitepix[x,y][2]))

    new_im = grey.copy()
    binarize(new_im, 150)
    return new_im
def binarize(im, thresh=120):
    assert 0 < thresh < 255
    assert im.mode == 'L'
    w, h = im.size
    for y in xrange(0, h):
        for x in xrange(0, w):
            if im.getpixel((x,y)) < thresh:
                im.putpixel((x,y), 0)
            else:
                im.putpixel((x,y), 255)
# 获取12306验证码
def get_img():
    resp = urllib.urlopen(pic_url)
    raw = resp.read()
    tmp_jpg = tempfile.NamedTemporaryFile(prefix="12306_").name + ".jpg"
    with open(tmp_jpg, 'wb') as fp:
        fp.write(raw)
    im = Image.open(tmp_jpg)
    try:
        os.remove(tmp_jpg)
    except OSError:
        pass
    return im
#用PIL 扣出图片
def get_sub_img(im, x, y):
    assert 0 <= x <= 3
    assert 0 <= y <= 2
    WITH = HEIGHT = 68
    left = 5 + (67 + 5) * x
    top = 41 + (67 + 5) * y
    right = left + 67
    bottom = top + 67

    return im.crop((left, top, right, bottom))

# 调用百度webservice
def baidu_stu_lookup(im):
    url = "http://stu.baidu.com/n/image?fr=html5&needRawImageUrl=true&id=WU_FILE_0&name=233.png&type=image%2Fpng&lastModifiedDate=Mon+Mar+16+2015+20%3A49%3A11+GMT%2B0800+(CST)&size="
    tmp_jpg = tempfile.NamedTemporaryFile(prefix="fuck12306_").name + ".png"
    im.save(tmp_jpg)
    raw = open(tmp_jpg, 'rb').read()
    try:
        os.remove(tmp_jpg)
    except OSError:
        pass
    url = url + str(len(raw))
    req = urllib2.Request(url, raw, {'Content-Type':'image/png', 'User-Agent':UA})
    resp = urllib2.urlopen(req)
    resp_url = resp.read()

    url = "http://stu.baidu.com/n/searchpc?queryImageUrl=" + urllib.quote(resp_url)

    req = urllib2.Request(url, headers={'User-Agent':UA})
    resp = urllib2.urlopen(req)

    html = resp.read()

    return baidu_stu_html_extract(html)
# 正则处理返回结果
def baidu_stu_html_extract(html):
    #pattern = re.compile(r'<script type="text/javascript">(.*?)</script>', re.DOTALL | re.MULTILINE)
    pattern = re.compile(r"keywords:'(.*?)'")
    matches = pattern.findall(html)
    if not matches:
        return '[UNKNOWN]'
    json_str = matches[0]

    json_str = json_str.replace('\\x22', '"').replace('\\\\', '\\')

    print json_str

    result = [item['keyword'] for item in json.loads(json_str)]

    return '|'.join(result) if result else '[UNKNOWN]'

if __name__ == '__main__':
    im = get_img()
    print "start"
    #im = Image.open("C://test.jpg") # 测试图片
    print 'OCR Question:', ocr_question_extract(im)
    for y in range(2):
        for x in range(4):
            im2 = get_sub_img(im, x, y)
            result = baidu_stu_lookup(im2)
            print (y,x), result
    print "end"