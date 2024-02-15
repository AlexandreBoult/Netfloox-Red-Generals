from imports2 import *

def reset_stgs():
    colors = [(44, 46, 63),(74, 77, 105),(99, 0, 192),(124, 0, 240),(31, 34, 45)]
    autocomplete=["aaa","bbb","lol"]
    return colors,autocomplete

reset_stgs()
model_instances={}
nb_instances=0

app = Flask(__name__)
sass.compile(dirname=('asset', 'static'))
app.config['UPLOAD_FOLDER'] = 'settings'

@app.route("/", methods=['POST','GET'])
def index():
    global dflt_par,model_instances,nb_instances

    if request.method == 'GET':
        colors,autocomplete=reset_stgs()
        id_vault=str(nb_instances)
        nb_instances+=1
        vault=(colors,autocomplete,1,id_vault)

    elif request.method == 'POST':
        colors,autocomplete,id_vault=ast.literal_eval(request.form.get('vault'))[:3]
        #truc = request.form.get('test')
        #if truc == "1" :
            #do truc
        vault=(colors,autocomplete,0,id_vault)
        return render_template("index.html",vault=vault)
    
    return render_template("index.html",vault=vault)

if __name__ == "__main__":
    app.run(threaded=True)