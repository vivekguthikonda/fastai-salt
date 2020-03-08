import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.staticfiles import StaticFiles
from google_drive_downloader import GoogleDriveDownloader as gdd
import PIL
import base64

export_file_gid = '1J8SEtCgZ8nnpN5HW1dTs-JSqP_kzVTc8'
export_file_name = 'export.pkl'

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

def acc(input, target):
    target = target.squeeze(1)
    return (input.argmax(dim=1) == target).float().mean()

async def download_file(gid, dest):
    if dest.exists(): return
    gdd.download_file_from_google_drive(file_id = gid, dest_path = dest)


async def setup_learner():
    await download_file(export_file_gid, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    outputs = learn.predict(img.resize(128))
    im = image2np(outputs[1])
    resp_bytes = BytesIO()
    PIL.Image.fromarray((im*200).astype('uint8')).save(resp_bytes, format='png')
    img_str = base64.b64encode(resp_bytes.getvalue()).decode()
    img_str = "data:image/png;base64," + img_str
    return JSONResponse({'image': img_str})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
