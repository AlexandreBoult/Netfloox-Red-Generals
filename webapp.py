from imports2 import *

def reset_stgs():
    colors = [(44, 46, 63),(74, 77, 105),(99, 0, 192),(124, 0, 240),(31, 34, 45)]
    autocomplete=["aaa","bbb","lol"]
    return colors,autocomplete

reset_stgs()

nb_instances=0

app = Flask(__name__)
sass.compile(dirname=('asset', 'static'))
app.config['UPLOAD_FOLDER'] = 'settings'

@app.route("/", methods=['POST','GET'])
def index():
    global dflt_par,nb_instances

    if request.method == 'GET':
        colors,autocomplete=reset_stgs()
        id_vault=str(nb_instances)
        nb_instances+=1
        vault=(colors,autocomplete,1,[],id_vault)

    elif request.method == 'POST':
        colors,autocomplete,first_time,profile,id_vault=ast.literal_eval(request.form.get('vault'))
        quiz = request.form.get('test1')
        
        if quiz == "1" :
            return redirect(url_for('quiz'), code=307)
        vault=(colors,autocomplete,0,profile,id_vault)
        return render_template("index.html",vault=vault)
    
    return render_template("index.html",vault=vault)

if __name__ == "__main__":
    app.run(threaded=True)

@app.route("/quiz", methods=['POST','GET'])
def quiz():
    global dflt_par,nb_instances
    selection=fetch_random_movie_info(5)
    colors,autocomplete,first_time,profile,id_vault=ast.literal_eval(request.form.get('vault'))
    vault=(colors,autocomplete,0,profile,id_vault)
    return render_template("quiz.html",vault=vault)
