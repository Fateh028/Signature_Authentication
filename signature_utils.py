import base64
import io
import json
from typing import List, Tuple, Dict
import traceback

import cv2
import numpy as np
from PIL import Image

try:
    from scipy.spatial.distance import euclidean
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using numpy alternatives")

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: skimage not available, SSIM will be disabled")

from config import SIG_WIDTH, SIG_HEIGHT, ORB_SIM_THRESHOLD



def decode_base64_image(data_url: str) -> np.ndarray:
    """
    Decode a 'data:image/png;base64,...' URL into a BGR OpenCV image.
    """
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def preprocess_signature(img_bgr: np.ndarray) -> np.ndarray:
    """
    Advanced preprocessing with noise reduction and normalization.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold for better binarization
    thr = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to clean up the signature
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    
    resized = cv2.resize(thr, (SIG_WIDTH, SIG_HEIGHT), interpolation=cv2.INTER_AREA)
    norm = resized.astype("float32") / 255.0
    return norm


def extract_features(img: np.ndarray) -> Dict:
    """
    Extract comprehensive features from signature.
    """
    features = {}
    img_uint8 = (img * 255).astype("uint8")
    
    # 1. Contour features
    contours, _ = cv2.findContours(img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        features['contour_area'] = float(cv2.contourArea(main_contour))
        features['contour_perimeter'] = float(cv2.arcLength(main_contour, True))
        features['num_contours'] = len(contours)
        
        # Hu moments
        moments = cv2.moments(main_contour)
        if moments['m00'] != 0:
            hu_moments = cv2.HuMoments(moments).flatten()
            features['hu_moments'] = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        else:
            features['hu_moments'] = np.zeros(7)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(main_contour)
        features['aspect_ratio'] = float(w / h if h > 0 else 0)
        features['bbox_area'] = float(w * h)
        
        # Convex hull
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        features['solidity'] = float(features['contour_area'] / hull_area if hull_area > 0 else 0)
        
        # Approximate polygon
        epsilon = 0.01 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        features['num_vertices'] = len(approx)
        
    else:
        features['contour_area'] = 0.0
        features['contour_perimeter'] = 0.0
        features['num_contours'] = 0
        features['hu_moments'] = np.zeros(7)
        features['aspect_ratio'] = 0.0
        features['bbox_area'] = 0.0
        features['solidity'] = 0.0
        features['num_vertices'] = 0
    
    # 2. ORB keypoints
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(img_uint8, None)
    features['orb_kp'] = kp
    features['orb_des'] = des
    features['orb_count'] = len(kp) if kp else 0
    
    # 3. Pixel distribution
    features['pixel_density'] = float(np.sum(img) / (SIG_WIDTH * SIG_HEIGHT))
    features['pixel_count'] = int(np.sum(img > 0))
    
    # Projections
    h_proj = np.sum(img, axis=1)
    v_proj = np.sum(img, axis=0)
    features['h_projection'] = h_proj.astype(float)
    features['v_projection'] = v_proj.astype(float)
    
    # 4. Grid features (4x4 grid)
    grid_size = 4
    h_step = SIG_HEIGHT // grid_size
    w_step = SIG_WIDTH // grid_size
    grid_features = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            cell = img[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            grid_features.append(float(np.sum(cell) / cell.size))
    
    features['grid_densities'] = np.array(grid_features, dtype=float)
    
    return features


def orb_similarity(features1: Dict, features2: Dict) -> float:
    """
    ORB keypoint matching with quality checks.
    """
    try:
        des1 = features1['orb_des']
        des2 = features2['orb_des']
        
        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            return 0.0
        
        # Check keypoint count similarity
        kp_ratio = min(features1['orb_count'], features2['orb_count']) / max(features1['orb_count'], features2['orb_count'])
        if kp_ratio < 0.15:
            return 0.0
        
        # BFMatcher with ratio test
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 5:
            return 0.0
        
        # Calculate similarity
        match_ratio = len(good_matches) / min(len(des1), len(des2))
        avg_distance = np.mean([m.distance for m in good_matches])
        distance_score = max(0.0, 1.0 - (avg_distance / 80.0))
        
        score = (match_ratio * 0.6 + distance_score * 0.4)
        return float(max(0.0, min(1.0, score)))
        
    except Exception as e:
        print(f"ORB matching error: {e}")
        return 0.0


def compare_shapes(features1: Dict, features2: Dict) -> float:
    """
    Compare shape-based features.
    """
    try:
        scores = []
        
        # Hu moments
        hu1 = features1['hu_moments']
        hu2 = features2['hu_moments']
        if hu1 is not None and hu2 is not None:
            hu_dist = safe_euclidean(hu1, hu2)
            hu_sim = 1.0 / (1.0 + hu_dist)
            scores.append(hu_sim * 0.3)
        
        # Contour properties
        if features1['contour_area'] > 0 and features2['contour_area'] > 0:
            area_ratio = min(features1['contour_area'], features2['contour_area']) / max(features1['contour_area'], features2['contour_area'])
            scores.append(area_ratio * 0.2)
            
            perim_ratio = min(features1['contour_perimeter'], features2['contour_perimeter']) / max(features1['contour_perimeter'], features2['contour_perimeter'])
            scores.append(perim_ratio * 0.15)
            
            solidity_diff = abs(features1['solidity'] - features2['solidity'])
            solidity_sim = max(0.0, 1.0 - solidity_diff)
            scores.append(solidity_sim * 0.15)
        
        # Aspect ratio
        aspect_diff = abs(features1['aspect_ratio'] - features2['aspect_ratio'])
        aspect_sim = max(0.0, 1.0 - min(aspect_diff, 1.0))
        scores.append(aspect_sim * 0.2)
        
        return float(sum(scores)) if scores else 0.0
        
    except Exception as e:
        print(f"Shape comparison error: {e}")
        return 0.0


def compare_distributions(features1: Dict, features2: Dict) -> float:
    """
    Compare pixel distribution features.p
    """
    try:
        scores = []
        
        # Projection correlation
        h1 = features1['h_projection']
        h2 = features2['h_projection']
        v1 = features1['v_projection']
        v2 = features2['v_projection']
        
        if np.sum(h1) > 0 and np.sum(h2) > 0:
            h1_norm = h1 / (np.sum(h1) + 1e-10)
            h2_norm = h2 / (np.sum(h2) + 1e-10)
            h_corr = safe_pearsonr(h1_norm, h2_norm)
            if not np.isnan(h_corr):
                scores.append((h_corr + 1) / 2 * 0.25)
        
        if np.sum(v1) > 0 and np.sum(v2) > 0:
            v1_norm = v1 / (np.sum(v1) + 1e-10)
            v2_norm = v2 / (np.sum(v2) + 1e-10)
            v_corr = safe_pearsonr(v1_norm, v2_norm)
            if not np.isnan(v_corr):
                scores.append((v_corr + 1) / 2 * 0.25)
        
        # Grid correlation
        g1 = features1['grid_densities']
        g2 = features2['grid_densities']
        if np.sum(g1) > 0 and np.sum(g2) > 0:
            g_corr = safe_pearsonr(g1, g2)
            if not np.isnan(g_corr):
                scores.append((g_corr + 1) / 2 * 0.3)
        
        # Pixel density
        density_diff = abs(features1['pixel_density'] - features2['pixel_density'])
        density_sim = max(0.0, 1.0 - density_diff * 2)
        scores.append(density_sim * 0.2)
        
        return float(sum(scores)) if scores else 0.0
        
    except Exception as e:
        print(f"Distribution comparison error: {e}")
        return 0.0


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute SSIM if available.
    """
    if not SKIMAGE_AVAILABLE:
        return 0.0
    
    try:
        img1_uint8 = (img1 * 255).astype("uint8")
        img2_uint8 = (img2 * 255).astype("uint8")
        return float(ssim(img1_uint8, img2_uint8))
    except Exception as e:
        print(f"SSIM error: {e}")
        return 0.0


def compare_signatures(data_url1: str, data_url2: str) -> Tuple[float, str]:
    """
    Compare two signatures using multiple methods.
    """
    try:
        print("Starting signature comparison...")
        
        img1 = decode_base64_image(data_url1)
        img2 = decode_base64_image(data_url2)
        
        pre1 = preprocess_signature(img1)
        pre2 = preprocess_signature(img2)
        
        # Extract features
        features1 = extract_features(pre1)
        features2 = extract_features(pre2)
        
        print(f"Features extracted - Pixels1: {features1['pixel_count']}, Pixels2: {features2['pixel_count']}")
        
        # Compare using multiple methods
        orb_score = orb_similarity(features1, features2)
        shape_score = compare_shapes(features1, features2)
        dist_score = compare_distributions(features1, features2)
        ssim_score = compute_ssim(pre1, pre2)
        
        print(f"Scores - ORB: {orb_score:.3f}, Shape: {shape_score:.3f}, Dist: {dist_score:.3f}, SSIM: {ssim_score:.3f}")
        
        # Weighted combination
        if SKIMAGE_AVAILABLE:
            final_score = (
                orb_score * 0.35 +
                shape_score * 0.30 +
                dist_score * 0.25 +
                ssim_score * 0.10
            )
        else:
            final_score = (
                orb_score * 0.40 +
                shape_score * 0.35 +
                dist_score * 0.25
            )
        
        print(f"Final similarity: {final_score:.3f}")
        return float(final_score), "multi_feature"
        
    except Exception as e:
        print(f"Error in compare_signatures: {e}")
        traceback.print_exc()
        return 0.0, "error"


def encode_template(preprocessed_img: np.ndarray) -> str:
    """
    Encode signature template for storage.
    """
    try:
        features = extract_features(preprocessed_img)
        
        # Convert numpy arrays to lists for JSON serialization
        template = {
            'image': preprocessed_img.astype("float32").flatten().tolist(),
            'hu_moments': features['hu_moments'].tolist(),
            'contour_area': float(features['contour_area']),
            'contour_perimeter': float(features['contour_perimeter']),
            'solidity': float(features['solidity']),
            'num_vertices': int(features['num_vertices']),
            'aspect_ratio': float(features['aspect_ratio']),
            'pixel_density': float(features['pixel_density']),
            'pixel_count': int(features['pixel_count']),
            'h_projection': features['h_projection'].tolist(),
            'v_projection': features['v_projection'].tolist(),
            'grid_densities': features['grid_densities'].tolist(),
            'orb_count': int(features['orb_count'])
        }
        
        return json.dumps(template)
    except Exception as e:
        print(f"Error encoding template: {e}")
        traceback.print_exc()
        raise


def decode_template(s: str) -> Tuple[np.ndarray, Dict]:
    """
    Decode stored template.
    """
    try:
        template = json.loads(s)
        img = np.array(template['image'], dtype='float32').reshape(SIG_HEIGHT, SIG_WIDTH)
        
        # Reconstruct features dict
        features = {
            'hu_moments': np.array(template['hu_moments'], dtype=float),
            'contour_area': float(template['contour_area']),
            'contour_perimeter': float(template['contour_perimeter']),
            'solidity': float(template['solidity']),
            'num_vertices': int(template['num_vertices']),
            'aspect_ratio': float(template['aspect_ratio']),
            'pixel_density': float(template['pixel_density']),
            'pixel_count': int(template['pixel_count']),
            'h_projection': np.array(template['h_projection'], dtype=float),
            'v_projection': np.array(template['v_projection'], dtype=float),
            'grid_densities': np.array(template['grid_densities'], dtype=float),
            'orb_count': int(template['orb_count']),
            'orb_des': None,  # Can't store ORB descriptors in JSON easily
            'orb_kp': None
        }
        
        return img, features
    except Exception as e:
        print(f"Error decoding template: {e}")
        traceback.print_exc()
        raise


def verify_signature_against_stored(
    new_data_url: str, stored_template_json: str
) -> Tuple[bool, float, str]:
    """
    Verify new signature against stored template.
    """
    try:
        print("Starting verification...")
        
        img_new = decode_base64_image(new_data_url)
        pre_new = preprocess_signature(img_new)
        
        # Extract features from new signature
        features_new = extract_features(pre_new)
        
        # Decode stored template
        stored_img, stored_features = decode_template(stored_template_json)
        
        print(f"Comparing - New pixels: {features_new['pixel_count']}, Stored pixels: {stored_features['pixel_count']}")
        
        # Compare (skip ORB for stored since we don't have descriptors)
        shape_score = compare_shapes(features_new, stored_features)
        dist_score = compare_distributions(features_new, stored_features)
        ssim_score = compute_ssim(pre_new, stored_img)
        
        # For new vs stored, recalculate ORB from stored image
        stored_features_full = extract_features(stored_img)
        orb_score = orb_similarity(features_new, stored_features_full)
        
        print(f"Verification scores - ORB: {orb_score:.3f}, Shape: {shape_score:.3f}, Dist: {dist_score:.3f}, SSIM: {ssim_score:.3f}")
        
        # Weighted combination
        if SKIMAGE_AVAILABLE:
            final_score = (
                orb_score * 0.35 +
                shape_score * 0.30 +
                dist_score * 0.25 +
                ssim_score * 0.10
            )
        else:
            final_score = (
                orb_score * 0.40 +
                shape_score * 0.35 +
                dist_score * 0.25
            )
        
        print(f"Final verification score: {final_score:.3f}, Threshold: {ORB_SIM_THRESHOLD}")
        
        ok = final_score >= ORB_SIM_THRESHOLD
        
        return ok, float(final_score), "multi_feature"
        
    except Exception as e:
        print(f"Error in verify_signature_against_stored: {e}")
        traceback.print_exc()
        return False, 0.0, "error"

def safe_euclidean(a, b):
    """Euclidean distance with fallback if scipy not available."""
    if SCIPY_AVAILABLE:
        return euclidean(a, b)
    else:
        return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))


def safe_pearsonr(a, b):
    """Pearson correlation with fallback if scipy not available."""
    if SCIPY_AVAILABLE:
        try:
            corr, _ = pearsonr(a, b)
            return corr
        except:
            return 0.0
    else:
        # Numpy alternative
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        corr = np.corrcoef(a, b)[0, 1]
        return 0.0 if np.isnan(corr) else corr

