import datetime
import itk
import lxml.etree as ET
import numpy as np
from pathlib import Path
from pymidas.common.libxml import ProjectXml, SubjectXml
import zlib
from typing import Optional
import SimpleITK as sitk
import skimage
import xarray as xr
from typing_extensions import Self

def midas_itk_to_sitk(image):
    """ """
    simple = sitk.GetImageFromArray(itk.GetArrayFromImage(image))
    simple.SetOrigin(tuple(image.GetOrigin()))
    simple.SetSpacing(tuple(image.GetSpacing()))
    simple.SetDirection(itk.GetArrayFromMatrix(image.GetDirection()).flatten())
    return simple

def slicer_array(image: itk.Image) -> np.ndarray:
    """ """
    return np.flipud(itk.GetArrayFromImage(image))

def midas_sitk_to_itk(simple):
    """ """
    image = itk.GetImageFromArray(sitk.GetArrayFromImage(simple))
    image.SetOrigin(simple.GetOrigin())
    image.SetSpacing(simple.GetSpacing())
    image.SetDirection(
        itk.GetMatrixFromArray(np.array(simple.GetDirection()).reshape(3, 3))
    )
    return image

def parse_data_type(data_type):
    """ """
    if data_type.lower() == "integer":
        return np.int16, itk.Image[itk.SS, 3]
    elif data_type.lower() == "byte":
        return np.uint8, itk.Image[itk.UC, 3]
    elif data_type.lower() == "float":
        return np.float32, itk.Image[itk.F, 3]
    elif data_type.lower() == "double":
        return np.float64, itk.Image[itk.D, 3]
    else:
        raise Exception(f"Invalid data type: {data_type}")
    
def cast_float(image):
    """ """
    caster = itk.CastImageFilter[image, itk.Image[itk.F, 3]].New()
    caster.SetInput(image)
    caster.Update()
    return caster.GetOutput()

def cross_product(xr, yr, zr, xc, yc, zc):
    """ """
    return [yr * zc - zr * yc, zr * xc - xr * zc, xr * yc - yr * xc]

def update_origin(x, y, z, xr, yr, zr, xc, yc, zc):
    """ """
    DCM = np.array(
        [
            [
                xr,
                yr,
                zr,
            ],
            [
                xc,
                yc,
                zc,
            ],
            cross_product(xr, yr, zr, xc, yc, zc),
        ],
        dtype=np.float64,
    )

    origin = np.array(
        [
            [x],
            [y],
            [z],
        ],
        dtype=np.float64,
    )

    new_origin = DCM @ origin

    return list(new_origin.reshape(-1))

def _register_routine(
    fixed, moving, learn_rate=4.0, stop=0.01, max_steps=200, log=False
):
    """ """

    def command_iteration(method):
        print(
            f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}"
        )

    fixed = midas_itk_to_sitk(fixed)
    moving = midas_itk_to_sitk(moving)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(learn_rate, stop, max_steps)
    # R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInitialTransform(sitk.VersorRigid3DTransform())
    R.SetInterpolator(sitk.sitkLinear)

    if log:
        R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    out = midas_sitk_to_itk(out)

    return out, outTx 

def register_resample(fixed, moving, Tx):
    """ """
    fixed = midas_itk_to_sitk(fixed)
    moving = midas_itk_to_sitk(moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(Tx)

    out = resampler.Execute(moving)

    out = midas_sitk_to_itk(out)

    return out  

def align_map(mapImage, t1Image):
    """ """
    # We use the identity transform to maintain the map's current orientation.
    identity = itk.IdentityTransform[itk.D, 3].New()
    identity.SetIdentity()

    # Set types.
    ImageFloatType = itk.Image[itk.F, 3]

    # Cast to float
    img = cast_float(mapImage)

    # Use the resample filter to resample the map in t1 space
    ResampleFilterType = itk.ResampleImageFilter[ImageFloatType, ImageFloatType]
    resampler = ResampleFilterType.New()
    resampler.SetTransform(identity)
    resampler.SetOutputParametersFromImage(t1Image)
    resampler.SetInput(img)
    resampler.Update()

    return resampler.GetOutput() 

class MidasNode:
    """ """

    id: Optional[str]
    node: ET._Element
    subject_xml: SubjectXml

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        self.id = None
        self.node = node
        self.subject_xml = subject_xml
        self.subject_path = subject_path

    def all_param(self):
        """ """
        # TODO: loop is redundant
        params = {}
        param_list = [
            (x.get("name"), x.get("value")) for x in self.node.xpath(f"./param")
        ]
        for p in param_list:
            name, value = p[0], p[1]
            params[name] = [x[1] for x in param_list if x[0] == name]
            if len(params[name]) == 1:
                params[name] = params[name][0]
        return params

    def param(self, name):
        """ """
        return self.subject_xml.get_parameter_given_id(self.id, name)


class MidasFrame(MidasNode):
    """ """

    path: Path

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Frame_ID']/@value")[0]
        self.path = self.subject_path / str(
			self.subject_xml.get_file_path_given_id(self.id)
		).replace("\\", "/")

    def load(self):
        """ """
        offset = int(self.param("Byte_Offset"))

        data_type = self.param("Data_Representation")
        dtype, ImageInputType = parse_data_type(data_type)

        px = float(self.param("Image_Position_X"))
        py = float(self.param("Image_Position_Y"))
        pz = float(self.param("Image_Position_Z"))
        dx = int(self.param("Spatial_Points_1"))
        dy = int(self.param("Spatial_Points_2"))
        dz = int(self.param("Spatial_Points_3"))
        sx = float(self.param("Pixel_Spacing_1"))
        sy = float(self.param("Pixel_Spacing_2"))
        sz = float(self.param("Pixel_Spacing_3"))
        oxr = float(self.param("Image_Orientation_Xr"))
        oyr = float(self.param("Image_Orientation_Yr"))
        ozr = float(self.param("Image_Orientation_Zr"))
        oxc = float(self.param("Image_Orientation_Xc"))
        oyc = float(self.param("Image_Orientation_Yc"))
        ozc = float(self.param("Image_Orientation_Zc"))

        array = np.fromfile(self.path, dtype=dtype, count=dx * dy * dz, offset=offset)
        array = array.reshape(dz, dy, dx)
        image = itk.GetImageFromArray(array)

        image.SetSpacing([sx, sy, sz])
        image.SetOrigin(update_origin(px, py, pz, oxr, oyr, ozr, oxc, oyc, ozc))
        image.SetDirection(
            np.array(
                [
                    [
                        oxr,
                        oyr,
                        ozr,
                    ],
                    [
                        oxc,
                        oyc,
                        ozc,
                    ],
                    cross_product(oxr, oyr, ozr, oxc, oyc, ozc),
                ],
                dtype=np.float64,
            )
        )

        if data_type != "float":
            image = cast_float(image)

        return array, image


class MidasInput(MidasNode):
    """ """

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Input_ID']/@value")[0]
        self.output_data_id = self.param("Output_Data_ID")

    def data(self):
        """ """
        return MidasData(
            self.node.xpath(
                f"../data[./param[@name='Data_ID' and @value='{self.output_data_id}']]"
            )[0],
            self.subject_xml,
            self.subject_path,
        )


class MidasData(MidasNode):
    """ """

    path: Path

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Data_ID']/@value")[0]
        self.path = self.subject_path / str(
			self.subject_xml.get_file_path_given_id(self.id)
		).replace("\\", "/")

    def all_frame(self):
        """ """
        return [
            MidasFrame(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./frame")
        ]

    def frame(self, frame_type):
        """ """
        return MidasFrame(
            self.node.xpath(
                f"./frame[./param[@name='Frame_Type' and @value='{frame_type}']]"
            )[0],
            self.subject_xml,
            self.subject_path,
        )

    def load_all_frames(self):
        """ """
        frame_type_list = self.node.xpath(f"./frame/param[@name='Frame_Type']/@value")
        frames = {frame_type: self.load_frame(frame_type)} # type: ignore
        return frames

    def load_frame(self, frame_type=None):
        """ """
        if frame_type == None:
            frame = MidasFrame(
                self.node.xpath(f"./frame")[0], self.subject_xml, self.subject_path
            )
        else:
            frame = MidasFrame(
                self.node.xpath(
                    f"./frame[param[@name='Frame_Type' and @value='{frame_type}']"
                )[0],
                self.subject_xml,
                self.subject_path,
            )
        return frame.load()

    def load_image(self):
        """ """
        data_type = self.param("Data_Representation")
        dtype, ImageInputType = parse_data_type(data_type)

        px = float(self.param("Image_Position_X"))
        py = float(self.param("Image_Position_Y"))
        pz = float(self.param("Image_Position_Z"))
        dx = int(self.param("Spatial_Points_1"))
        dy = int(self.param("Spatial_Points_2"))
        dz = int(self.param("Spatial_Points_3"))
        sx = float(self.param("Pixel_Spacing_1"))
        sy = float(self.param("Pixel_Spacing_2"))
        sz = float(self.param("Pixel_Spacing_3"))
        oxr = float(self.param("Image_Orientation_Xr"))
        oyr = float(self.param("Image_Orientation_Yr"))
        ozr = float(self.param("Image_Orientation_Zr"))
        oxc = float(self.param("Image_Orientation_Xc"))
        oyc = float(self.param("Image_Orientation_Yc"))
        ozc = float(self.param("Image_Orientation_Zc"))

        array = np.fromfile(self.path, dtype=dtype, count=dx * dy * dz)
        array = array.reshape(dz, dy, dx)
        image = itk.GetImageFromArray(array)

        image.SetSpacing([sx, sy, sz])
        image.SetOrigin(update_origin(px, py, pz, oxr, oyr, ozr, oxc, oyc, ozc))
        image.SetDirection(
            np.array(
                [
                    [
                        oxr,
                        oyr,
                        ozr,
                    ],
                    [
                        oxc,
                        oyc,
                        ozc,
                    ],
                    cross_product(oxr, oyr, ozr, oxc, oyc, ozc),
                ],
                dtype=np.float64,
            )
        )

        if data_type != "float":
            image = cast_float(image)

        return array, image

    def load_spectra(self):
        """ """
        if self.param("Compression") and self.param("Compression").lower() == "zlib":
            with open(self.path, "rb") as f:
                spec = np.frombuffer(zlib.decompress(f.read()), dtype=np.float32)
        else:
            spec = np.fromfile(self.path, dtype=np.float32)

        spec = spec.reshape(
            int(self.param("Spatial_Points_3")),
            int(self.param("Spatial_Points_2")),
            int(self.param("Spatial_Points_1")),
            int(self.param("Spectral_Points_1")),
            2,
        )

        return spec[..., 0] + 1j * spec[..., 1]


class MidasProcess(MidasNode):
    """ """

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Process_ID']/@value")[0]

    def dataset(self, created_by):
        """ """
        if created_by == None:
            return MidasDataset(
                self.node.xpath(f"./dataset")[0], self.subject_xml, self.subject_path
            )
        else:
            return MidasDataset(
                self.node.xpath(
                    f"./dataset[./param[@name='Created_By' and @value='{created_by}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )

    def all_input(self):
        """ """
        return [
            MidasInput(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./input")
        ]

    def input(self, process_name):
        """ """
        return MidasInput(
            self.node.xpath(
                f"./input[./param[@name='Process_Name' and @value='{process_name}']]"
            )[0],
            self.subject_xml,
            self.subject_path,
        )

    def all_data(self):
        """ """
        return [
            MidasData(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./data")
        ]

    def data(self, created_by=None, frame_type=None):
        """ """
        if created_by == None and frame_type == None:
            return MidasData(
                self.node.xpath(f"./data")[0], self.subject_xml, self.subject_path
            )
        elif created_by:
            return MidasData(
                self.node.xpath(
                    f"./data[./param[@name='Created_By' and @value='{created_by}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )
        elif frame_type:
            return MidasData(
                self.node.xpath(
                    f"./data[./frame/param[@name='Frame_Type' and @value='{frame_type}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )

    def all_frame(self):
        """ """
        return [
            MidasFrame(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./data/frame")
        ]

    def frame(self, frame_type):
        """ """
        return MidasFrame(
            self.node.xpath(
                f"./data/frame[./param[@name='Frame_Type' and @value='{frame_type}']]"
            )[0],
            self.subject_xml,
            self.subject_path,
        )


class MidasDataset(MidasNode):
    """ """

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Dataset_ID']/@value")[0]

    def all_data(self):
        """ """
        return [
            MidasData(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./data")
        ]

    def data(self, created_by=None):
        """ """
        if created_by == None:
            return MidasData(
                self.node.xpath(f"./data")[0], self.subject_xml, self.subject_path
            )
        else:
            return MidasData(
                self.node.xpath(
                    f"./data[./param[@name='Created_By' and @value='{created_by}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )

    def all_frame(self):
        """ """
        return [
            MidasFrame(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./data/frame")
        ]

    def frame(self, frame_type):
        """ """
        return MidasFrame(
            self.node.xpath(
                f"./data/frame[./param[@name='Frame_Type' and @value='{frame_type}']]"
            )[0],
            self.subject_xml,
            self.subject_path,
        )


class MidasSeries(MidasNode):
    """ """

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Series_ID']/@value")[0]

    def all_process(self):
        """ """
        return [
            MidasProcess(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./process")
        ]

    def process(self, label):
        """ """
        try:
            return MidasProcess(
                self.node.xpath(f"./process[./param[@name='Label' and @value='{label}']]")[
                    0
                ],
                self.subject_xml,
                self.subject_path,
            )
        except IndexError:
            return None

    def all_dataset(self):
        """ """
        return [
            MidasDataset(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./dataset")
        ]

    def dataset(self, label):
        """ """
        return MidasDataset(
            self.node.xpath(f"./dataset[./param[@name='Label' and @value='{label}']]")[
                0
            ],
            self.subject_xml,
            self.subject_path,
        )


class MidasStudy(MidasNode):
    """ """

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Study_ID']/@value")[0]
        self.date = self.param("Study_Date")
        self.subject_id = self.param("Subject_ID")

    def all_series(self):
        """ """
        return [
            MidasSeries(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./series")
        ]

    def series(self, label=None, series_id=None):
        """ """
        if label:
            # Search for a series with the specified label 
            nodes = self.node.xpath(f"./series[./param[@name='Label' and @value='{label}']]")
            if not nodes:
                raise Exception(f"Series with label '{label}' does not exist.")
            return MidasSeries(
                self.node.xpath(
                    f"./series[./param[@name='Label' and @value='{label}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )
        elif series_id:
            # Search for a series with the specified label 
            nodes = self.node.xpath(f"./series[./param[@name='Label' and @value='{series_id}']]")
            if not nodes:
                raise Exception(f"Series with label '{label}' does not exist.")
            return MidasSeries(
                self.node.xpath(
                    f"./series[./param[@name='Series_ID' and @value='{series_id}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )
        else:
            raise Exception(f"No label or series_id specified")

    def process(self, label):
        """ """
        return MidasProcess(
            self.node.xpath(f"./process[./param[@name='Label' and @value='{label}']]")[
                0
            ],
            self.subject_xml,
            self.subject_path,
        )

    def t1(self):
        """ """
        data = self.series("MRI_T1").process("Volume").data()
        return data.load_image()
    
    def t1pc(self):
        """ """
        data = self.series("MRI_T1_PC").process("Volume").data()
        return data.load_image()

    def flair(self):
        """ """
        data = self.series("MRI_FLAIR").process("Volume").data()
        return data.load_image()

    def ref(self):
        """ """
        data = self.series("SI_Ref").process("Maps").data()
        return data.load_image()

    def brain_mask(self):
        """ """
        frame = self.series("SI_Ref").process("Maps").frame("Mask_Brain")
        return frame.load()

    def lipid_mask(self):
        """ """
        frame = self.series("SI_Ref").process("Maps").frame("Mask_Lipid")
        return frame.load()

    def si(self):
        """ """
        return self.series("SI").process("Spectral").data().load_spectra()

    def siref(self):
        """ """
        return self.series("SI_Ref").process("Spectral").data().load_spectra()

    def spectral_sampling(self, node):
        """ """
        hz_per_ppm = float(node.param("Precession_Frequency"))
        spec_pts = int(node.param("Spectral_Points_1"))
        freq_offset = float(node.param("Frequency_Offset"))
        chem_shift_ref = float(node.param("Chemical_Shift_Reference"))
        spec_width = float(node.param("Spectral_Width_1"))
        hz_per_pt = spec_width / spec_pts
        ppm_range = spec_width / hz_per_ppm
        ppm_per_pt = ppm_range / spec_pts
        center_ppm = chem_shift_ref + freq_offset
        left_edge_ppm = center_ppm + (ppm_range / 2)

        return dict(
            hz_per_ppm=hz_per_ppm,
            spec_pts=spec_pts,
            freq_offset=freq_offset,
            chem_shift_ref=chem_shift_ref,
            spec_width=spec_width,
            hz_per_pt=hz_per_pt,
            ppm_range=ppm_range,
            ppm_per_pt=ppm_per_pt,
            center_ppm=center_ppm,
            left_edge_ppm=left_edge_ppm,
        )

    def si_sampling(self):
        """ """
        data = self.series("SI").process("Spectral").data()
        return self.spectral_sampling(data)

    def siref_sampling(self):
        """ """
        data = self.series("SI_Ref").process("Spectral").data()
        return self.spectral_sampling(data)

    def fitt(self):
        """ """
        if not self.series("SI").process("Spectral_FitBase"):
            return None
        return self.series("SI").process("Spectral_FitBase").data().load_spectra()

    def fitt_baseline(self):
        """ """
        if not self.series("SI").process("Spectral_BL"):
            return None
        return self.series("SI").process("Spectral_BL").data().load_spectra()

    def qmap(self):
        """ """
        frame = self.series("SI").process("Maps").data("QMaps").frame("Quality_Map")
        return frame.load()

    def t2star(self):
        """ """
        frame = self.series("SI_Ref").process("Maps").frame("T2Star_Map")
        return frame.load()

    def segmentation(self, label):
        """ """
        data = self.process("MRI_SEG").dataset("MriSeg").data(label)
        return data.load_frame()


class MidasSubject(MidasNode):
    """ """

    def __init__(self, subject_xml: Path):
        """ """
        subject_xml = Path(subject_xml)
        super().__init__(
            ET.parse(
                subject_xml, parser=ET.XMLParser(remove_blank_text=True)
            ).getroot(),
            SubjectXml(str(subject_xml)),
            subject_xml.parent,
        )
        self.id = self.node.xpath(f"./param[@name='Subject_ID']/@value")[0]

    def all_study(self):
        """ """
        return [
            MidasStudy(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./study")
        ]

    def study(self, subject_name, study_date, study_time=None):
        """ """
        if study_time == None:
            return MidasStudy(
                self.node.xpath(
                    f"./study[./param[@name='Study_Date' and @value='{study_date}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )
        else:
            return MidasStudy(
                self.node.xpath(
                    f"./study"
                    f"[./param[@name='Study_Date' and @value='{study_date}']]"
                    f"[./param[@name='Study_Time' and @value='{study_time}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )


class MidasProject:
    """ """

    def __init__(self, project_xml: Path):
        """ """
        project_xml = Path(project_xml)
        self.node = ET.parse(
            project_xml, parser=ET.XMLParser(remove_blank_text=True)
        ).getroot()
        # self.project_xml = ProjectXml(str(project_xml))
        self.project_xml = project_xml
        self.path = project_xml.parent
        self.name = self.node.xpath(f"./param[@name='Project_Name']/@value")[0]

    def all_subject(self):
        """ """
        subject_nodes = self.node.xpath(f"./Subject")
        subject_list = []
        for node in subject_nodes:
            subject_dir = node.xpath(f"./param[@name='Subject_Directory']/@value")[0]
            subject_xml = self.path / subject_dir / "subject.xml"
            subject_list.append(MidasSubject(subject_xml))
        return subject_list

    def subject(self, subject_id):
        """ """
        return MidasSubject(self.path / subject_id / "subject.xml")
    

class OnixObject:
    """
    Base class for Onix objects.
    """
    def __init__(self):
        self._silent = True

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def _silence(self):
        self._silent = True

    def _unsilence(self):
        self._silent = False


class OnixVolume(OnixObject):
    """
    Class that represents a multi-dimensional image.

    Attributes
    ----------
    array : np.ndarray
        The raw image data as a numpy array.
    image : itk.Image
        The image data as an itk.Image.
    is_mask : bool  
        Whether the volume is a mask or not.
    _silent : bool
        Whether to raise warnings or not.

    Methods
    -------
    register(fixed, tx)
        Register data to the fixed image using the specified transform.
    align(fixed)
        Align data to the coordinate system of the fixed image.
    connected_component()
        Find the largest connected component in the mask.
    slicer_array()
        Get the image data as a numpy array for the slicer.
    flip_x()
        Flip the volume along the x-axis.
    """
    # ^ sphnix documentation

    array: np.ndarray
    image: itk.Image
    is_mask: bool

    def __init__(self, array: np.ndarray, image: itk.Image, is_mask: bool = False):
        super().__init__()
        self.array = array
        self.image = image
        self.is_mask = is_mask

    def register(self, fixed: Self, tx) -> Self:
        """
        Register data to the fixed image using the specified transform.
        """
        image = register_resample(fixed.image, self.image, tx)
        return OnixVolume(self.array, image, self.is_mask)

    def align(self, fixed: Self) -> Self:
        """
        Align data to the coordinate system of the fixed image.
        """
        image = align_map(self.image, fixed.image)
        return OnixVolume(self.array, image, self.is_mask)

    def connected_component(self) -> Self:
        """
        Find the largest connected component in the mask.
        """
        if self.is_mask:
            #f = skimage.morphology.isotropic_closing
            from skimage.morphology import isotropic_closing
            f = isotropic_closing
            img_bw = f(self.array, 2)
            labels = skimage.measure.label(
                img_bw, 
                return_num=False,
            )
            maxCC_nobcg = labels == np.argmax(
                np.bincount(labels.flat, weights=img_bw.flat)
            )
            array = maxCC_nobcg.astype(np.uint8)
            image = itk.GetImageFromArray(array)
            image.SetOrigin(self.image.GetOrigin())
            image.SetSpacing(self.image.GetSpacing())
            image.SetDirection(self.image.GetDirection())
            return OnixVolume(array, image, is_mask=True)
        else:
            if self._silent:
                return None
            else:
                raise Exception(f"Cannot apply connected component to non-mask")

    def slicer_array(self) -> np.ndarray:
        """
        Get the image data as a numpy array for the slicer.
        """
        return slicer_array(self.image)

    def flip_x(self) -> Self:
        """
        Flip the volume along the x-axis.
        """
        array = self.array[:, :, ::-1]
        image = self.image
        flipFilter = itk.FlipImageFilter[image].New()
        flipFilter.SetInput(image)
        flipAxes = (True, False, False)
        flipFilter.SetFlipAxes(flipAxes)
        flipFilter.Update()
        image = flipFilter.GetOutput()
        return OnixVolume(array, image, self.is_mask)


class NNFitDataset(OnixObject):
    """
    Represents the data from an nnfit process.

    Attributes
    ----------
    study : MidasStudy
        The study node from MIDAS.
    process : MidasProcess
        The nnfit process node from MIDAS.
    data : MidasData
        The nnfit data node from MIDAS.
    xr_data : MidasData
        The xarray data node from MIDAS.
    """
    # ^ sphnix documentation

    study: MidasStudy
    process: MidasProcess
    data: MidasData
    xr_data: MidasData

    def __init__(self, study: MidasStudy, og=True):
        self.study = study
        self.process = self.study.series("SI").process("nnfit")
        self.data = self.process.data("nnfit")
        self.xr_data = self.process.data("xarray")

        if og:
            self.load_og()
            self._og = True
        else:
            try:
                self.load_og()
                self._og = True
            except Exception as e:
                print(f"\nWARNING: unable to load OG data\n{e}\n")
                self._og = False

        self.metabolite = self.open_ds().metabolite.data
        self.ratio = self.open_ds().ratio.data
        self.ppm = self.open_ds().ppm.data

    def load_og(self):
        """ """
        maps_process = self.study.series("SI").process("Maps")
        try:
            self.og_data = maps_process.data("NNFIT")
        except IndexError:
            # If "NNFIT" wasn't found, try "nnfit"
            self.og_data = maps_process.data("nnfit")

        self.og_cho = OnixVolume(*self.og_data.frame("nnfit_CHO_Area").load())
        self.og_cr = OnixVolume(*self.og_data.frame("nnfit_CR_Area").load())
        self.og_naa = OnixVolume(*self.og_data.frame("nnfit_NAA_Area").load())
        self.og_cho_naa = OnixVolume(*self.og_data.frame("nnfit_CHO/NAA").load())

        self.og_maps = ['og_cho', 'og_cr', 'og_naa', 'og_cho_naa']

        nnfit_dir = self.study.subject_path / "nnfit"

        spec_file = nnfit_dir / self.og_data.param("nnfit_spectrum_file")
        self.og_spec = np.fromfile(spec_file, dtype=np.float32).reshape(
            int(self.og_data.param("Spatial_Points_3")),
            int(self.og_data.param("Spatial_Points_2")),
            int(self.og_data.param("Spatial_Points_1")),
            512,
        )
        
        base_file = nnfit_dir / self.og_data.param("nnfit_baseline_file")
        self.og_base = np.fromfile(base_file, dtype=np.float32).reshape(
            int(self.og_data.param("Spatial_Points_3")),
            int(self.og_data.param("Spatial_Points_2")),
            int(self.og_data.param("Spatial_Points_1")),
            512,
        )

    def load_og_map(self, label) -> (np.ndarray, itk.Image): # type: ignore
        """ """
        if label == "og_cho":
            return OnixVolume(self.og_cho.array, self.og_cho.image)
        elif label == "og_cr":
            return OnixVolume(self.og_cr.array, self.og_cr.image)
        elif label == "og_naa":
            return OnixVolume(self.og_naa.array, self.og_naa.image)
        elif label == "og_cho_naa":
            return OnixVolume(self.og_cho_naa.array, self.og_cho_naa.image)
        else:
            raise Exception(f"{label} is not an OG nnfit map")

    def open_ds(self):
        """
        Open the xarray dataset.
        """
        return xr.open_zarr(self.xr_data.path, decode_times=False).sel(frame="Original")

    def ndarray_to_itk(self, array: np.ndarray) -> itk.Image:
        """
        Convert a numpy array to an itk image.

        Assumes that metadata is stored in Midas data object.
        """
        # Extract the metadata for the image
        px = float(self.data.param("Image_Position_X"))
        py = float(self.data.param("Image_Position_Y"))
        pz = float(self.data.param("Image_Position_Z"))
        dx = int(self.data.param("Spatial_Points_1"))
        dy = int(self.data.param("Spatial_Points_2"))
        dz = int(self.data.param("Spatial_Points_3"))
        sx = float(self.data.param("Pixel_Spacing_1"))
        sy = float(self.data.param("Pixel_Spacing_2"))
        sz = float(self.data.param("Pixel_Spacing_3"))
        oxr = float(self.data.param("Image_Orientation_Xr"))
        oyr = float(self.data.param("Image_Orientation_Yr"))
        ozr = float(self.data.param("Image_Orientation_Zr"))
        oxc = float(self.data.param("Image_Orientation_Xc"))
        oyc = float(self.data.param("Image_Orientation_Yc"))
        ozc = float(self.data.param("Image_Orientation_Zc"))

        # Pass the raw data to an itk image
        image = itk.GetImageFromArray(array)

        # Update image with the metadata
        image.SetSpacing([sx, sy, sz])
        image.SetOrigin(update_origin(px, py, pz, oxr, oyr, ozr, oxc, oyc, ozc))
        image.SetDirection(
            np.array(
                [
                    [oxr, oyr, ozr],
                    [oxc, oyc, ozc],
                    cross_product(oxr, oyr, ozr, oxc, oyc, ozc),
                ],
                dtype=np.float64,
            )
        )

        # Cast image to float, since image operations are float
        if array.dtype != np.float32:
            image = cast_float(image)

        return image

    def load_area(self, metabolite: str) -> (np.ndarray, itk.Image):  # type: ignore
        """
        Load metabolite area (i.e. amplitude) map.
        """
        ds = self.open_ds()
        array = ds.sel(metabolite=metabolite).areas.data.compute()
        image = self.ndarray_to_itk(array)
        return array, image

    def load_shift(self, metabolite: str) -> (np.ndarray, itk.Image):  # type: ignore
        """
        Load metabolite-specific frequency shift map.
        """
        ds = self.open_ds()
        array = (
            ds.dw.data.compute() + ds.sel(metabolite=metabolite).shifts.data.compute()
        )
        image = self.ndarray_to_itk(array)
        return array, image

    def shift(self, x, y, z, option="points"):
        """
        Load the frequency shift (based on NAA) at the specified coordinates.
        """
        ds = self.open_ds().sel(z=z, y=y, x=x)
        freq_shift = (
            ds.dw.data.compute() + ds.sel(metabolite="naa").shifts.data.compute()
        )

        if option == "points":
            hz_per_pt = self.study.si_sampling().get("hz_per_pt")
            return int(freq_shift / (2 * np.pi) / hz_per_pt)
        elif option == "ppm":
            hz_per_ppm = self.study.si_sampling().get("hz_per_ppm")
            return freq_shift / (2 * np.pi) / hz_per_ppm
        elif option == "hz":
            return freq_shift / (2 * np.pi)
        else: 
            # rad/s
            return freq_shift

    def phase(self, x, y, z):
        """
        Load the phase at the specified coordinates.
        """
        ds = self.open_ds()
        return ds.phi0.sel(z=z, y=y, x=x).data.compute()

    def load_map(self, label) -> (np.ndarray, itk.Image):  # type: ignore
        """
        Load the specified map.
        """
        ds = self.open_ds()
        array = ds.get(label).data.compute()
        image = self.ndarray_to_itk(array)
        return array, image

    def load_ratio(self, ratio) -> (np.ndarray, itk.Image):  # type: ignore
        """
        Load the specified ratio map.
        """
        ds = self.open_ds()
        array = ds.sel(ratio=ratio).ratios.data.compute()
        image = self.ndarray_to_itk(array)
        return array, image

    def load_spectra(self) -> np.ndarray:
        """
        Load the spectral data.
        """
        ds = self.open_ds()
        if isinstance(ds.spectrum.data, np.ndarray):
            return ds.spectrum.data  # Already a NumPy array
        else:
            return ds.spectrum.data.compute()  # Compute if it's a Dask array
        #return ds.spectrum.data.compute()

    def load_peaks(self):
        """
        Load the peaks fit (i.e. metabolite signal fit).
        """
        ds = self.open_ds()
        if isinstance(ds.peaks.data, np.ndarray):
            return ds.peaks.data  # Already a NumPy array
        else:
            return ds.peaks.data.compute()  # Compute if it's a Dask array
        #return ds.peaks.data.compute()

    def load_baseline(self):
        """
        Load the baseline fit.
        """
        ds = self.open_ds()
        if isinstance(ds.baseline.data, np.ndarray):
            return ds.baseline.data  # Already a NumPy array
        else:
            return ds.baseline.data.compute()  # Compute if it's a Dask array
        #return ds.baseline.data.compute()
