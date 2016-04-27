12306_captcha

TODO:文字和图片的组合验证（学习andel）

环境

安装: python下的图片处理库PIL

安装 OCR库google 的pytesseract

百度服务请求：http://stu.baidu.com/n/image?fr=html5&needRawImageUrl=true&id=WU_FILE_0&name=233.png&type=image%2Fpng&lastModifiedDate=Mon+Mar+16+2015+20%3A49%3A11+GMT%2B0800+(CST)&size=

额外
'''
urllib2.Request(url + str(len("图片")), "图片", {'Content-Type':'image/png', 'User-Agent':UA})
'''
百度识别12306，屏蔽了小尺寸的图片识别（100px以下的）使用ImageMagick放大尺寸。 切图和放大图片，需要安装一下ImageMagick，确保能在命令行打出convert命令。
'''
//放大图片尺寸 
def resizeImg() 
{ 
	arrargs = ["-resize", "440", "./test.jpg", "./new_test.jpg"]; 
	command = "convert"+ arrargs[0] + arrargs[1]+arrargs[2]+arrargs[3] os.system(command) 
}
'''