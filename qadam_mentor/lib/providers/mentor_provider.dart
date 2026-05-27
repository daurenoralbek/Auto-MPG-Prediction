import 'package:flutter/material.dart';
import '../models/mentor_model.dart';
import '../services/firestore_service.dart';

class MentorProvider extends ChangeNotifier {
  final FirestoreService _service = FirestoreService();

  List<MentorModel> _allMentors = [];
  List<MentorModel> _filteredMentors = [];
  String _searchQuery = '';
  String? _selectedCategory;
  bool _isLoading = false;
  String? _error;

  List<MentorModel> get mentors =>
      _filteredMentors.isEmpty && _searchQuery.isEmpty && _selectedCategory == null
          ? _allMentors
          : _filteredMentors;
  bool get isLoading => _isLoading;
  String? get error => _error;
  String get searchQuery => _searchQuery;
  String? get selectedCategory => _selectedCategory;

  void listenToMentors() {
    _isLoading = true;
    notifyListeners();
    _service.getMentorsStream().listen(
      (list) {
        _allMentors = list;
        _applyFilters();
        _isLoading = false;
        notifyListeners();
      },
      onError: (e) {
        _error = 'Менторларды жүктеу мүмкін болмады';
        _isLoading = false;
        notifyListeners();
      },
    );
  }

  void search(String query) {
    _searchQuery = query.toLowerCase();
    _applyFilters();
    notifyListeners();
  }

  void filterByCategory(String? categoryId) {
    _selectedCategory = categoryId;
    _applyFilters();
    notifyListeners();
  }

  void clearFilters() {
    _searchQuery = '';
    _selectedCategory = null;
    _filteredMentors = [];
    notifyListeners();
  }

  void _applyFilters() {
    var list = List<MentorModel>.from(_allMentors);
    if (_selectedCategory != null && _selectedCategory!.isNotEmpty) {
      list = list
          .where((m) => m.categories.contains(_selectedCategory))
          .toList();
    }
    if (_searchQuery.isNotEmpty) {
      list = list
          .where((m) =>
              m.name.toLowerCase().contains(_searchQuery) ||
              m.specialty.toLowerCase().contains(_searchQuery) ||
              m.bio.toLowerCase().contains(_searchQuery))
          .toList();
    }
    _filteredMentors = list;
  }
}
