import os
import conf as cfg
import cv2


def rm_ds(array: list):
    try:
        array.remove('.DS_Store')
    except Exception as e:
        print(e)


def extract_centralframe(imgpath: str, H, W):
    img = cv2.imread(imgpath)
    h, w, c = img.shape
    centerh, centerw = int(h/2), int(w/2)
    dh, dw = int(H/2), int(W/2)
    return img[centerh-dh: centerh+dh, centerw-dw:centerw+dw, :]


def all_iframes(srcpath: str, trgpath: str):
    folders = os.listdir(srcpath)
    folders = rm_ds(folders)
    for folder in folders:
        folderpath = os.path.join(srcpath, folder)
        trgfolder = os.path.join(trgpath, folder)
        iframes = os.listdir(folderpath)
        iframes = rm_ds(iframes)
        for iframe in iframes:
            iframepath = os.path.join(folderpath. iframe)
            iframetrgpath = os.path.join(trgfolder, iframe)
            crop = extract_centralframe(iframepath)
            cv2.imwrite(filename=iframetrgpath, img=crop)










def main():
    print(cfg.paths['root'])



if __name__ == '__main__':
    main()