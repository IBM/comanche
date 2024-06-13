#include "SQLParser.h"
#include <sstream>
#include <unordered_map>
#include <cctype>
#include <stdexcept>
#include <algorithm> // Add this include for std::all_of

// Define a mapping from string to TokenType
static const std::unordered_map<std::string, TokenType> tokenMap = {
    {"SELECT", TokenType::SELECT},
    {"FROM", TokenType::FROM},
    {"WHERE", TokenType::WHERE},
    {"AND", TokenType::AND},
    {"OR", TokenType::OR},
    {"BETWEEN", TokenType::BETWEEN}, // Added BETWEEN operator
    {"=", TokenType::OPERATOR},
    {">", TokenType::OPERATOR},
    {">=", TokenType::OPERATOR},
    {"<", TokenType::OPERATOR},
    {"<=", TokenType::OPERATOR},
    {"!=", TokenType::OPERATOR}
};

TokenType SQLParser::getTokenType(const std::string& token) {
    auto it = tokenMap.find(token);
    if (it != tokenMap.end()) {
        return it->second;
    } else if (isColumnName(token)) {
        return TokenType::COLUMN;
    } else if (isLiteral(token)) {
        return TokenType::LITERAL;
    } else {
        return TokenType::UNKNOWN;
    }
}

bool SQLParser::isColumnName(const std::string& token) {
    // Implement logic to determine if a token is a column name
    // For simplicity, let's assume column names do not contain spaces and are alphanumeric
    return std::all_of(token.begin(), token.end(), [](char c) {
        return std::isalnum(c) || c == '_';
    });
}

bool SQLParser::isLiteral(const std::string& token) {
    // Implement logic to determine if a token is a literal
    // For simplicity, let's assume literals are either numeric or quoted strings
    if (token.empty()) return false;
    if (token.front() == '\'' && token.back() == '\'') return true; // Quoted string literal
    return std::all_of(token.begin(), token.end(), ::isdigit);      // Numeric literal
}

std::vector<Token> SQLParser::parse(const std::string& sql) {
    std::vector<Token> tokens;
    std::istringstream stream(sql);
    std::string word;

    while (stream >> word) {
        TokenType type = getTokenType(word);
        tokens.push_back({type, word});
    }

    return tokens;
}

std::string SQLParser::tokenTypeToString(TokenType type) {
    switch (type) {
        case TokenType::SELECT: return "SELECT";
        case TokenType::FROM: return "FROM";
        case TokenType::WHERE: return "WHERE";
        case TokenType::COLUMN: return "COLUMN";
        case TokenType::LITERAL: return "LITERAL";
        case TokenType::OPERATOR: return "OPERATOR";
        case TokenType::AND: return "AND";
        case TokenType::OR: return "OR";
        case TokenType::BETWEEN: return "BETWEEN";
        default: return "UNKNOWN";
    }
}
