import 'package:flutter/material.dart';
import '../services/firestore_service.dart';

class FavoritesProvider extends ChangeNotifier {
  final FirestoreService _service = FirestoreService();

  Set<String> _favoriteIds = {};

  Set<String> get favoriteIds => _favoriteIds;

  bool isFavorite(String mentorId) => _favoriteIds.contains(mentorId);

  void listenToFavorites(String userId) {
    _service.getFavoriteMentorIds(userId).listen((ids) {
      _favoriteIds = ids.toSet();
      notifyListeners();
    });
  }

  Future<void> toggle(String userId, String mentorId) async {
    if (_favoriteIds.contains(mentorId)) {
      _favoriteIds.remove(mentorId);
      notifyListeners();
      await _service.removeFavorite(userId, mentorId);
    } else {
      _favoriteIds.add(mentorId);
      notifyListeners();
      await _service.addFavorite(userId, mentorId);
    }
  }

  void clear() {
    _favoriteIds = {};
    notifyListeners();
  }
}
