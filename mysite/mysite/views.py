# from django.http import HttpResponse
# from django.shortcuts import render
# # import pandas as pd
# import numpy as np
# import pandas as pd
# import matplotlib as pl
# import matplotlib.pyplot as plt
# pl.use('Agg')
# import io
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# # import matplotlib.pyplot as plt
# # import seaborn as sns
# import pandas_datareader as pdr
# import quandl
# from datetime import datetime
# from django.http import HttpResponse
# from matplotlib import pylab
# from pylab import *
# from . import stock









# def index(request):
#     return render(request,'index.html')








# def fun(request):
#     a = c()
#     djtext = request.GET.get('text','default')
#     dj = request.GET.get('text1','default')
#     context = {'post':params}
#     return render(request,'Chart.html',context)



# def datasets(request):
#     djtext = request.GET.get('text','default',)


#     df = pdr.get_data_yahoo(djtext,start=datetime.datetime(2006, 10, 1),end=datetime.datetime(2012, 1, 1))
#     # params={'dj':html}
#     # params ={dj:df.to_json()}
#     df.reset_index().to_html('templates/new.html',index=False,border=0,col_space=100)
#     return render(request,'new.html')
#     # return HttpResponse(df.to_html(classes='table table-bordered'))

	
# 	# return render(request,'datasets.html',params)
# 	# return render(request,'datasets.html',{"dj":mma})



# # import PIL, PIL.Image, StringIO

# def getimage(request):
#     # Construct the graph
#     x = arange(0, 2*pi, 0.01)
#     s = cos(x)**2
#     plot(x, s)

#     xlabel('xlabel(X)')
#     ylabel('ylabel(Y)')
#     title('Simple Graph!')
#     grid(True)

#     # Store image in a string buffer
#     buffer = StringIO.StringIO()
#     canvas = pylab.get_current_fig_manager().canvas
#     canvas.draw()
#     pilImage = PIL.Image.fromstring("RGB", canvas.get_width_height(), canvas.tostring_rgb())
#     pilImage.save(buffer, "PNG")
#     pylab.close()

#     # Send buffer in a http response the the browser with the mime type image/png set
#     return HttpResponse(buffer.getvalue(), mimetype="image/png")


# def getNums(request):
#         n = np.array([2, 3, 4])
#         name1 = "name-" + str(n[1])
#         return HttpResponse('{ "name":"' + name1 + '", "age":31, "city":"New York" }')



# def getData(request):
	
#     samp = np.random.randint(100, 600, size=(4, 5))
#     df = pd.DataFrame(samp, index=['avi', 'dani', 'rina', 'dina'],
#                       columns=['Jan', 'Feb', 'Mar', 'Apr', 'May'])
#     return HttpResponse(df.to_html(classes='table table-bordered'))


# # def getimage(request):
# #     x = np.arange(0, 2 * np.pi, 0.01)
# #     s = np.cos(x) ** 2
# #     plt.plot(x, s)

# #     plt.xlabel('xlabel(X)')
# #     plt.ylabel('ylabel(Y)')
# #     plt.title('Simple Graph!')
# #     plt.grid(True)

# #     response = HttpResponse(content_type="image/jpeg")
# #     plt.savefig(response, format="png")
# #     return render(response,'textgrab.html')







# # def getimage(request):
# #     fig =pl.figure.Figure()
# #     canvas = FigureCanvas(fig)
# #     x = np.arange(-2, 1.5, .01)
# #     y = np.sin(np.exp(2 * x))
# #     plt.plot(x, y)
# #     buf = io.BytesIO()
# #     plt.savefig(buf, format='png')
# #     plt.close(fig)
# #     response =HttpResponse(content_type='image/png')
# #     canvas.print_png(response)
# #     return response









# def textgrab(request):
# 	# djtext = request.GET.get('text','default')
# 	# params={'dj':djtext}
# 	# dj=getimage(request)
# 	# params={'djimg':dj}
	
# 	# return render(request,'textgrab.html',params)
# 	return render(request,'Chart.html')
